# 环境搭建与运行

**何时读**：首次使用本 skill；或运行报错说缺依赖、模型下载 404、torch/CUDA 不可用。

## 依赖清单

```
paddleocr==3.7.0
paddlex==3.7.2
transformers==5.18.0
imageio-ffmpeg==0.6.0
numpy
onnxruntime
RapidFuzz
rapidocr
pyclipper
yt-dlp
```

（同仓库根的 `requirements.txt` 是机器可读版本。）

要点：

- **不装 `paddlepaddle`。** 走的是 `transformers` 推理后端，模型跑在 PyTorch 上，不需要 PaddlePaddle 运行时。
- `rapidocr` / `onnxruntime` 只在需要跑 CPU 对照路线时才用到；主线（`paddleocr-tf`）不需要它们。
- `yt-dlp` 用于从链接下载视频——脚本本身不下载。

## 环境：conda 环境 `skill-ocr`

**不建 venv，也不用 `--system-site-packages`。** 建一个独立 conda 环境，自带一份 CUDA 版
torch，与系统里的其他环境互不影响：

```bash
conda create -n skill-ocr python=3.13 -y
conda run -n skill-ocr pip install -r requirements.txt
conda run -n skill-ocr pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

下文用 `<python>` 指代该环境的解释器。装完用 `pip check` 复核依赖自洽。

要点：

- **必须显式装 torch 的 cu128 版**。`requirements.txt` 里没有 torch；装 `transformers`
  时它可能拉来 CPU 版，必须用官方 cu128 索引覆盖，才能跑在 GPU 上。
- **torch 版本必须与显卡架构匹配**。torch 2.11+cu128 起才自带真实可用的
  sm_120 kernel（Blackwell / RTX 50 系）。升级/降级前先确认新版本仍支持你的架构。
- 走 `transformers` 推理后端，模型跑在 PyTorch 上，**不需要 PaddlePaddle 运行时**。

> 曾经的写法是 venv + `--system-site-packages` 继承外部 torch。已弃用：
> venv 不可迁移（`Scripts/*.exe` 写死绝对路径），且把环境放进 skill 目录会污染仓库。
> 现在统一用 conda 环境，路径不在 skill 文档里出现。

## 两个必需的环境变量

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
PADDLE_PDX_MODEL_SOURCE=huggingface
```

原因（实测教训 4）：默认的 aistudio / modelscope 模型源对 `PP-OCRv5_mobile_det_safetensors` 返回 404，HF 源才拿得到。脚本内部已用 `os.environ.setdefault` 兜底设了一次，但**显式在命令行前缀里给一遍更稳**（也避免被外部已有值覆盖）。

## 模型与缓存位置

```
~/.paddlex/official_models/            # PP-OCRv5/v6 模型（合计约 200MB）
```

典型内容：`PP-OCRv5_mobile_det_safetensors`（检测，仅 14MB）、`PP-OCRv6_medium_det_safetensors`、`PP-OCRv6_medium_rec_safetensors`（识别）。

- **首次运行会自动下载**，之后复用缓存；缓存可整目录删除，下次会重新下。
- **缓存与 venv 都不是交付物**，不要把它们当结果报给用户。

## 两条路线的取舍

| 路线 | 后端 | det 单帧 | 全片（716s/60fps） | 用途 |
|---|---|---|---|---|
| `paddleocr-tf`（默认） | transformers + torch GPU | **24 ms** | **约 20 分钟** | 主线，正式提取 |
| `rapidocr` | onnxruntime CPU | 约 1.5 s | 约 18 小时 | 只在 GPU 路线不可用时用于极小样本；**不要拿来跑全片** |

只有在这两种情况下才考虑 CPU 路线：GPU 依赖装不上、或显卡架构不被 torch 支持。也只有在 GPU 路线确实失败时，才值得去尝试 Intel 核显的 openvino 后端（未实测）。

## ffmpeg

脚本通过 `imageio-ffmpeg` 使用其自带的 ffmpeg 二进制，**无需系统安装 ffmpeg**。`imageio-ffmpeg` 已列在依赖清单里。

## 从零到跑通

```bash
# 1) 环境
conda create -n skill-ocr python=3.13 -y
<python> -m pip install -r requirements.txt
<python> -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) 小样验证 ROI
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PADDLE_PDX_MODEL_SOURCE=huggingface \
<python> scripts/hardsub_extract.py "<视频>" \
  --out "<dir>/probe" --seconds 120 --keep-frames --source-url "<URL>"

# 3) 核对 probe/_frames/*.png 后，跑全片
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PADDLE_PDX_MODEL_SOURCE=huggingface \
<python> scripts/hardsub_extract.py "<视频>" \
  --out "<dir>" --source-url "<URL>"
```
