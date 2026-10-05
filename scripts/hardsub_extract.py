#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hardsub_extract —— 中文视频硬字幕 OCR 提取（可复用工具）

pipeline: 逐帧解码(crop 字幕区) → 逐帧文本检测(det) → 连续帧投票切段
          → 段内几何变化触发 OCR(rec) + 0.75s 兜底 → rapidfuzz 归并 → SRT

依赖与运行方式见同目录上级的 references/environment-setup.md；
参数含义与调参方向见 references/roi-and-tuning.md。

用法:
  python hardsub_extract.py <video> --out <dir> [选项]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# ---------------- 默认参数 ----------------
DEF_ROI_TOP = 0.78       # ROI 上边界（探测值：字幕 y 82%~96.6%）
DEF_ROI_BOT = 0.98       # ROI 下边界
DEF_VOTE_N = 3           # 连续 N 帧投票
DEF_VOTE_K = 2           # N 帧中 ≥K 帧检出才算有字幕
DEF_SIM = 80             # rapidfuzz Levenshtein ratio 归并阈值
DEF_MERGE_GAP = 0.35     # 归并最大时间间隙（秒）
DEF_MIN_DUR = 0.30       # 最短字幕时长（秒）
DEF_CONF = 0.75          # OCR 置信度过滤


def log(msg=''):
    print(msg, flush=True)


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(video):
    ff = get_ffmpeg()
    r = subprocess.run([ff, '-i', video], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    import re
    w = h = 0
    fps = 30.0
    dur = 0.0
    for line in r.stderr.splitlines():
        if 'Stream #0' in line and 'Video:' in line:
            m = re.search(r'(\d{3,5})x(\d{3,5})', line)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
            m = re.search(r'([\d.]+)\s*tbr', line)
            if m:
                fps = float(m.group(1))
        if 'Duration:' in line:
            m = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', line)
            if m:
                dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return w, h, fps, dur


class FrameReader:
    """ffmpeg 管道逐帧读取器（-vf crop 只输出 ROI 行，避免整帧字节流错位）"""

    def __init__(self, video, width, height, roi_top=None, roi_bot=None):
        ff = get_ffmpeg()
        self.full_h = height
        self.y0 = int(height * roi_top) if roi_top else 0
        self.y1 = int(height * roi_bot) if roi_bot else height
        self.h = self.y1 - self.y0
        self.w = width
        self.frame_size = width * self.h * 3
        crop = f'crop={width}:{self.h}:0:{self.y0}'
        cmd = [ff, '-nostdin', '-i', video, '-vf', crop,
               '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-']
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     bufsize=self.frame_size * 4)
        self.idx = -1

    def read(self):
        buf = self.proc.stdout.read(self.frame_size)
        if not buf or len(buf) < self.frame_size:
            return None
        self.idx += 1
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((self.h, self.w, 3))
        return self.idx, arr

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.terminate()
        except Exception:
            pass


def make_engine(route):
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'huggingface')
    if route == 'paddleocr-tf':
        from paddleocr import TextDetection
        det = TextDetection(model_name='PP-OCRv5_mobile_det',
                            engine='transformers', device='gpu:0')
        return det
    elif route == 'rapidocr':
        from rapidocr import RapidOCR
        import logging
        logging.disable(logging.CRITICAL)
        return RapidOCR()
    raise ValueError(route)


def det_boxes(eng, route, img):
    """返回 (n_boxes, polys)。polys 为 None 之外的原始框列表"""
    if route == 'paddleocr-tf':
        res = eng.predict(img)
        for r in res:
            d = r.json if hasattr(r, 'json') else r
            d = d.get('res', d) if isinstance(d, dict) else d
            polys = d.get('dt_polys', []) or []
            return len(polys), [np.asarray(p) for p in polys]
        return 0, []
    else:
        out = eng(img)
        boxes = out.boxes
        if boxes is None:
            return 0, []
        boxes = [np.asarray(b) for b in boxes]
        return len(boxes), boxes


def union_box(polys):
    """所有框的并集 (x0,y0,x1,y1)"""
    if not polys:
        return None
    a = np.concatenate(polys, axis=0)
    return (float(a[:, 0].min()), float(a[:, 1].min()),
            float(a[:, 0].max()), float(a[:, 1].max()))


def box_changed(b1, b2, tol=6.0):
    """两个并集框是否显著不同（容差像素）"""
    if b1 is None or b2 is None:
        return b1 is not b2
    return any(abs(a - b) > tol for a, b in zip(b1, b2))


def ocr_middle(eng, route, img):
    """对稳定期中间帧跑完整 OCR，返回 (text, conf) 或 None"""
    if route == 'paddleocr-tf':
        from paddleocr import PaddleOCR
        global _rec_engine
        if '_rec_engine' not in globals():
            _rec_engine = PaddleOCR(
                engine='transformers', device='gpu:0',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        res = _rec_engine.predict(img)
        texts = []
        for r in res:
            d = r.json if hasattr(r, 'json') else r
            d = d.get('res', d) if isinstance(d, dict) else d
            t = d.get('rec_texts', []) or []
            s = d.get('rec_scores', []) or []
            texts = [(tt, float(ss)) for tt, ss in zip(t, s) if ss >= 0.5]
        if not texts:
            return None
        return ' '.join(t for t, _ in texts), max(s for _, s in texts)
    else:
        out = eng(img)
        txts = out.txts or ()
        scores = out.scores or ()
        pairs = [(t, float(s)) for t, s in zip(txts, scores) if s >= 0.5]
        if not pairs:
            return None
        return ' '.join(t for t, _ in pairs), max(s for _, s in pairs)


def main():
    ap = argparse.ArgumentParser(
        prog='hardsub_extract',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python hardsub_extract.py <视频.mp4> --out result/
  python hardsub_extract.py <视频.mp4> --out result/ --seconds 120   # 前2分钟验证
  python hardsub_extract.py <视频.mp4> --out result/ --route paddleocr-tf""")
    ap.add_argument('video', help='输入视频路径')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--route', default='paddleocr-tf',
                    choices=['paddleocr-tf', 'rapidocr'],
                    help='检测/识别后端（默认 paddleocr-tf，GPU；rapidocr 为 CPU 慢速对照）')
    ap.add_argument('--source-url', default='', help='写入 SRT 首行的来源链接')
    ap.add_argument('--roi-top', type=float, default=DEF_ROI_TOP)
    ap.add_argument('--roi-bot', type=float, default=DEF_ROI_BOT)
    ap.add_argument('--vote-n', type=int, default=DEF_VOTE_N)
    ap.add_argument('--vote-k', type=int, default=DEF_VOTE_K)
    ap.add_argument('--sim', type=int, default=DEF_SIM)
    ap.add_argument('--merge-gap', type=float, default=DEF_MERGE_GAP)
    ap.add_argument('--min-dur', type=float, default=DEF_MIN_DUR)
    ap.add_argument('--conf', type=float, default=DEF_CONF)
    ap.add_argument('--seconds', type=float, default=0, help='只处理前 N 秒（0=全片）')
    ap.add_argument('--keep-frames', action='store_true', help='保留每段中间帧截图')
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    W, H, fps, dur = probe_video(args.video)
    if not W:
        log('✗ 无法读取视频信息')
        sys.exit(6)
    log(f'▶ 视频: {args.video}')
    log(f'  {W}x{H} @ {fps:.1f}fps  时长 {dur:.1f}s')
    log(f'  ROI: {args.roi_top:.0%}~{args.roi_bot:.0%}  投票: {args.vote_n}中{args.vote_k}  路线: {args.route}')
    log()

    eng = make_engine(args.route)
    log('  引擎已装载')
    log()

    # ---------- 阶段1: 逐帧 det（记录每帧几何并集） ----------
    log('── 阶段1: 逐帧检测 ──')
    reader = FrameReader(args.video, W, H, args.roi_top, args.roi_bot)
    has_sub = []       # 每帧 bool
    geoms = []         # 每帧并集框 (x0,y0,x1,y1) 或 None
    n = 0
    t_det = 0.0
    while True:
        r = reader.read()
        if r is None:
            break
        idx, roi = r
        if args.seconds and idx / fps > args.seconds:
            break
        td = time.time()
        nboxes, polys = det_boxes(eng, args.route, roi)
        t_det += time.time() - td
        has_sub.append(nboxes > 0)
        geoms.append(union_box(polys) if nboxes > 0 else None)
        n += 1
        if n % 3000 == 0:
            el = time.time() - t0
            log(f'  {n} 帧 / det耗时 {t_det:.0f}s / 总 {el:.0f}s / 剩余约 {el/n*(dur-n/fps) if n else 0:.0f}s')
    reader.close()
    log(f'  共 {n} 帧, det 总耗时 {t_det:.0f}s')
    log()

    # ---------- 阶段2: 投票平滑 + 切段 ----------
    log('── 阶段2: 投票平滑 + 切段 ──')
    smoothed = vote_smooth(has_sub, args.vote_n, args.vote_k)
    segments = find_segments(smoothed, args.min_dur, fps)
    log(f'  切出 {len(segments)} 段')
    for i, (s, e) in enumerate(segments[:12]):
        log(f'    #{i+1} {s/fps:8.2f}s ~ {e/fps:8.2f}s')
    if len(segments) > 12:
        log(f'    ... 共 {len(segments)} 段')
    log()

    # ---------- 阶段3: 段内几何变化跟踪 + 触发式 OCR ----------
    log('── 阶段3: 段内 OCR（几何变化触发 + 0.75s 兜底） ──')
    # 对每个段: 从段起点开始, 遇到"几何变化"的帧或距上次 OCR 超过 0.75s, 就 OCR 一次
    ocr_jobs = []   # (frame_idx,)
    for (s, e) in segments:
        last_geom = None
        last_ocr_idx = -10**9
        for f in range(s, e + 1):
            g = geoms[f]
            if last_geom is None:
                ocr_jobs.append(f)
                last_ocr_idx = f
                last_geom = g
                continue
            # 几何变化 → 新字幕(或双行增减) → OCR
            if g is not None and box_changed(g, last_geom, tol=6.0):
                ocr_jobs.append(f)
                last_ocr_idx = f
                last_geom = g
            # 兜底: 每 0.75s 强制一次(应对文字变了几何没变的情况)
            elif f - last_ocr_idx >= int(fps * 0.75):
                ocr_jobs.append(f)
                last_ocr_idx = f
    log(f'  需 OCR {len(ocr_jobs)} 帧')
    reader = FrameReader(args.video, W, H, args.roi_top, args.roi_bot)
    ocr_results = {}
    if args.keep_frames:
        fd = os.path.join(args.out, '_frames')
        os.makedirs(fd, exist_ok=True)
    import cv2
    job_set = set(ocr_jobs)
    while job_set:
        r = reader.read()
        if r is None:
            break
        idx, roi = r
        if idx in job_set:
            res = ocr_middle(eng, args.route, roi)
            if res:
                ocr_results[idx] = res
            if args.keep_frames:
                cv2.imwrite(os.path.join(fd, f'f{idx:06d}.png'), roi)
            job_set.discard(idx)
    reader.close()
    got = len(ocr_results)
    log(f'  OCR 完成: {got}/{len(ocr_jobs)} 帧有文本')
    log()

    # ---------- 阶段4: 由 OCR 帧序列构建条目 + 归并 ----------
    log('── 阶段4: 归并 + SRT ──')
    # 每个OCR帧承载文本的时间范围: 从上一个OCR帧之后到下一个OCR帧(同段内)
    entries = []
    for (s, e) in segments:
        seg_jobs = [f for f in ocr_jobs if s <= f <= e]
        seg_texts = []
        for j in seg_jobs:
            res = ocr_results.get(j)
            if not res:
                continue
            text, conf = res
            if conf < args.conf:
                continue
            seg_texts.append((j, text.strip()))
        if not seg_texts:
            continue
        # 每个文本的 start = 该帧时间, end = 下一个不同文本帧的时间(或段尾+1帧)
        for k, (j, text) in enumerate(seg_texts):
            start_t = j / fps
            if k + 1 < len(seg_texts):
                end_t = seg_texts[k + 1][0] / fps
            else:
                end_t = (e + 1) / fps
            entries.append({'start': start_t, 'end': end_t, 'text': text})
    # 归并相邻同文本
    merged = merge_entries(entries, args)
    log(f'  原始 {len(entries)} 条 → 归并后 {len(merged)} 条')

    srt_path = os.path.join(args.out, 'subtitles.srt')
    write_srt(srt_path, merged, args.source_url)
    log(f'  SRT → {srt_path}')

    manifest = {
        'video': os.path.abspath(args.video),
        'route': args.route,
        'fps': fps, 'duration': dur,
        'frames_processed': n,
        'det_time_sec': round(t_det, 1),
        'segments': len(segments),
        'merged': len(merged),
        'params': {k: v for k, v in vars(args).items()},
        'elapsed_sec': round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, '_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log()
    log(f'✓ 完成。总耗时 {(time.time()-t0)/60:.1f} 分钟')


def vote_smooth(flags, n, k):
    out = []
    c = 0
    for i, v in enumerate(flags):
        c += 1 if v else 0
        if i >= n:
            c -= 1 if flags[i - n] else 0
        out.append(c >= k)
    return out


def find_segments(smoothed, min_dur, fps):
    segs = []
    start = None
    for i, v in enumerate(smoothed):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) / fps >= min_dur:
                segs.append((start, i - 1))
            start = None
    if start is not None and (len(smoothed) - start) / fps >= min_dur:
        segs.append((start, len(smoothed) - 1))
    return segs


def merge_entries(entries, args):
    from rapidfuzz import fuzz
    if not entries:
        return []
    merged = [dict(entries[0])]
    for e in entries[1:]:
        last = merged[-1]
        gap = e['start'] - last['end']
        ratio = fuzz.ratio(last['text'], e['text'])
        # 同文本(或高相似)合并, 不论间隙; 相邻零间隙且相似度高也合并
        if ratio > args.sim and (gap <= args.merge_gap or gap <= 0.05):
            last['end'] = e['end']
            if len(e['text']) > len(last['text']):
                last['text'] = e['text']
        else:
            merged.append(dict(e))
    # 时长过滤 + 清理重叠
    for a, b in zip(merged, merged[1:]):
        if b['start'] < a['end']:
            a['end'] = b['start']
    return [m for m in merged if m['end'] - m['start'] >= args.min_dur * 0.5]


def fmt_ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def write_srt(path, entries, source_url):
    lines = []
    if source_url:
        lines.append(source_url)
        lines.append('')
    for i, e in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{fmt_ts(e['start'])} --> {fmt_ts(e['end'])}")
        lines.append(e['text'])
        lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
