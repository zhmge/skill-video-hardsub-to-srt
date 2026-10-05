# 故障排查

**何时读**：运行失败、卡住，或结果明显不对。

## 症状 → 原因 → 处置

| 症状 | 可能原因 | 处置 |
|---|---|---|
| 模型下载报 404 / `<Response [404]>` | 没设模型源（教训 4） | 确认 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` 与 `PADDLE_PDX_MODEL_SOURCE=huggingface` 都在命令行前缀里；HF 偶发 502，重试即可 |
| 报 torch/CUDA 不可用，或算子不支持 sm_120 | venv 没带 `--system-site-packages`，装了 CPU 版 torch | 重建 venv：`<base-python> -m venv --system-site-packages ...`；确认继承来的 torch 是 cu128+ 版本 |
| 阶段 1 打印的帧数远小于预期（例如 716s 的视频只有几千帧） | `crop` 被移除了，按 ROI 高度切整帧字节流（教训 2） | 确认 `FrameReader` 的 ffmpeg 命令带 `-vf crop=W:H:0:y0` |
| 小样 `_frames` 全空 | ROI 没对准字幕 | 见 `roi-and-tuning.md`，确认框确实罩住字幕 |
| 小样 `_frames` 把背景也框进来了 | ROI 开太大 | 收窄 `--roi-top` / `--roi-bot` |
| 结果条数明显偏少，整句丢失 | 只 OCR 了段中间帧（教训 3） | 确认几何变化触发 + 0.75s 兜底逻辑还在 |
| 条数偏多、大量碎片 | 归并太弱 | 提高 `--min-dur`，必要时提高 `--sim` |
| 纯 UI / 水印文字混进结果 | 置信度太低 | 提高 `--conf` |
| 繁体字偶发混入 | PP-OCR 已知特性 | 需要强简体就在 `.srt` 之后接 opencc 后处理；不要改脚本 |
| 检测速度始终是 1s/帧 且不回落 | GPU 路线没生效（在跑 CPU） | 检查 torch 是否为 CUDA 版、`device='gpu:0'` 是否可用；不要用首帧计时下结论（教训 5，先预热） |
| 管道挂起 / 卡在阶段 1 不动 | ffmpeg 子进程异常 | 确认视频文件可正常解码（可单独用 ffmpeg 试一下）；确认 `imageio-ffmpeg` 已装 |
| 全片跑太久（数小时） | 正常现象 | 60fps 1080p 逐帧 det 约 20 分钟/集是预期值；确实要提速，可考虑把 det 换成 onnxruntime-gpu 版（**未实测**，不保证精度一致） |

## 已知局限（如实告知用户，不要当成 bug 去修）

- **换视频必须重新确认 ROI**：默认值只对「白字黑描边、底部居中」的一类片子成立。
- **繁体偶发混入**：PP-OCR 系模型的已知特性；需要全简体时用 opencc 后处理。
- **叠在字幕区的游戏 UI 文字可能混入**：靠 `--conf 0.75` + 几何过滤能挡掉大部分，但并非 100%。
- **60fps 逐帧 det 是耗时大头**（约 23ms/帧）：这是保证不漏句的代价，不要为了快而去掉逐帧检测。
- **`.srt` 文件名固定为 `subtitles.srt`**：这是脚本行为；要按视频命名需由调用方重命名。

## 快速自检清单

跑全片之前，确认这几点都成立：

- [ ] 两个环境变量已设置；
- [ ] 已跑过 `--seconds 120 --keep-frames` 小样；
- [ ] `_frames/*.png` 里的字幕完整落在 ROI 内；
- [ ] `--out` 已明确指定（不是猜的路径）；
- [ ] `--source-url` 已传入；
- [ ] 已向用户报过预计时长。
