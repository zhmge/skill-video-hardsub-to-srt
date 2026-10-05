# video-hardsub-to-srt

从视频画面里 OCR 提取**硬字幕**（烧录进画面的字幕），输出带时间轴的 `.srt`。

适用于视频没有独立字幕轨、字幕是画面一部分的情况：游戏实况、影视切片、录屏课程、没有 CC 的老视频。

## 它做什么

```
逐帧解码(crop 字幕区) → 逐帧文本检测 → 连续帧投票切段
  → 段内几何变化触发 OCR + 0.75s 兜底 → rapidfuzz 归并去重 → SRT
```

识别用 PaddleOCR 3.x 的 `transformers` 推理后端（跑在 PyTorch 上，不需要 PaddlePaddle 运行时），可在支持 CUDA 的 GPU 上以约 24 ms/帧的速度逐帧检测。

## 用法

```bash
# 1) 环境（继承已有的 CUDA 版 torch）
<base-python> -m venv --system-site-packages .venv-hardsub
.venv-hardsub/Scripts/python.exe -m pip install -r requirements.txt

# 2) 先跑小样确认字幕区（ROI）
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PADDLE_PDX_MODEL_SOURCE=huggingface \
.venv-hardsub/Scripts/python.exe scripts/hardsub_extract.py "<视频>" \
  --out "<dir>/probe" --seconds 120 --keep-frames --source-url "<URL>"

# 3) 核对 probe/_frames/*.png 后跑全片
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PADDLE_PDX_MODEL_SOURCE=huggingface \
.venv-hardsub/Scripts/python.exe scripts/hardsub_extract.py "<视频>" \
  --out "<dir>" --source-url "<URL>"
```

产物：`<dir>/subtitles.srt`（第 1 行 = 来源链接，第 2 行留空，其后为 cues）、`<dir>/_manifest.json`。

## 目录

| 路径 | 内容 |
|---|---|
| `SKILL.md` | agent 入口：适用边界、工作流、六条实测教训、不可妥协项 |
| `scripts/hardsub_extract.py` | 唯一实现，可独立当 CLI 用 |
| `references/environment-setup.md` | venv / 依赖 / 模型源 / 路线取舍 |
| `references/pipeline-design.md` | 四阶段原理 + 六条教训的实测依据 |
| `references/roi-and-tuning.md` | ROI 确认 + 参数逐项说明 |
| `references/troubleshooting.md` | 症状 → 原因 → 处置 |

## 边界

- **不要**用于有独立字幕轨 / CC / AI 字幕的视频——直接取字幕轨，不要 OCR。
- **不要**用于把视频转成笔记、大纲或摘要——那是下游 skill 的事。
- **不要**用于识别画面里的界面文字、弹幕、水印、标牌。
- 本 skill 只做**识别**，不做翻译、润色或改写。

## License

MIT
