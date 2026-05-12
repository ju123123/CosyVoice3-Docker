# Fun-CosyVoice3 TTS Docker 部署说明

本项目提供基于 FastAPI 的 Fun-CosyVoice3-0.5B 语音合成服务，接口兼容 OpenAI `/v1/audio/speech`，并支持部分 vLLM-Omni Speech API 扩展参数。

## 功能

- Docker / Docker Compose 部署
- GPU 推理，支持 CUDA 12.1 基础镜像
- 可选 vLLM 加速
- OpenAI 兼容 TTS 接口
- vLLM-Omni 风格扩展参数：`task_type`、`stream`、`ref_audio`、`ref_text` 等
- 支持非流式音频输出：`wav`、`mp3`、`flac`、`pcm`、`aac`、`opus`
- 支持 `stream=true` 流式 PCM 输出
- `ref_text` 自动拼接 CosyVoice3 前缀：`You are a helpful assistant.<|endofprompt|>`

## 目录

```text
.
├── cosyvoice_server.py
├── download_model.py
├── requirements.txt
├── test_client.py
├── asset/
├── models/
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── docker-entrypoint.sh
    ├── build-docker-image.sh
    └── .env
```

## 环境要求

- Linux 主机
- NVIDIA GPU，建议显存 8GB 以上
- 已安装 NVIDIA 驱动
- 已安装 Docker Engine
- 已安装 NVIDIA Container Toolkit
- 已安装 Docker Compose v2

检查 GPU 容器能力：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

如果该命令不能正常显示 GPU 信息，请先修复 NVIDIA Container Toolkit 配置。

## 快速部署

### 1. 准备目录

```bash
mkdir -p models asset output cache/modelscope
```

项目已包含示例音频：

```text
asset/longyingcheng_man.wav
asset/longyingwan_woman.wav
asset/longyingmu_woman.wav
```

### 2. 配置 Docker 参数

编辑 [docker/.env](docker/.env)：

```env
HOST_PORT=10096
SERVER_PORT=10096
GPU_DEVICE_ID=0
MODEL_DIR=/app/models/Fun-CosyVoice3-0.5B
USE_VLLM=true
USE_FP16=false
OUTPUT_SAMPLE_RATE=24000
```

如果机器只有一张 GPU，通常设置：

```env
GPU_DEVICE_ID=0
```

如需输出 16kHz PCM：

```env
OUTPUT_SAMPLE_RATE=16000
```

### 3. 构建镜像

```bash
cd docker
docker compose build
```

也可以使用构建脚本：

```bash
cd docker
bash build-docker-image.sh
```

### 4. 下载模型

容器启动时会检查 `/app/models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`。如果模型不存在，服务会退出并提示下载。

推荐直接在容器里下载到挂载目录：

```bash
cd docker
docker compose run --rm --entrypoint bash cosyvoice3 -lc "cd /app && python download_model.py"
```

下载完成后，宿主机目录应包含：

```text
models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
```

### 5. 启动服务

```bash
cd docker
docker compose up -d
```

查看日志：

```bash
docker compose logs -f cosyvoice3
```

停止服务：

```bash
docker compose down
```

## 服务验证

健康检查：

```bash
curl http://localhost:10096/health
```

音色列表：

```bash
curl http://localhost:10096/v1/audio/voices
```

非流式 WAV：

```bash
curl -X POST http://localhost:10096/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，这是一段测试音频。",
    "task_type": "Base",
    "ref_audio": "asset/longyingwan_woman.wav",
    "ref_text": "我们将为全球城市的可持续发展贡献力量。",
    "response_format": "wav"
  }' \
  -o test.wav
```

流式 PCM：

```bash
curl -X POST http://localhost:10096/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，这是一段测试音频。",
    "task_type": "Base",
    "ref_audio": "asset/longyingwan_woman.wav",
    "ref_text": "我们将为全球城市的可持续发展贡献力量。",
    "response_format": "pcm",
    "stream": true
  }' \
  -o test.pcm
```

播放 PCM：

```bash
ffplay -f s16le -ar 24000 -ac 1 test.pcm
```

如果 `.env` 中设置 `OUTPUT_SAMPLE_RATE=16000`，播放命令改为：

```bash
ffplay -f s16le -ar 16000 -ac 1 test.pcm
```

## API

### `GET /health`

返回服务状态、模型采样率、输出采样率、可用音色等信息。

### `GET /v1/audio/voices`

返回预置音色列表。

响应示例：

```json
{
  "voices": ["longyingcheng", "longyingwan", "longyingmu"],
  "uploaded_voices": []
}
```

### `POST /v1/audio/speech`

请求体为 JSON。

#### OpenAI 兼容字段

| 参数 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `model` | string | 否 | 兼容字段，当前不切换模型 |
| `input` | string | 是 | 要合成的文本 |
| `voice` | string | 否 | `task_type=CustomVoice` 时使用预置音色 |
| `response_format` | string | 否 | `wav`、`mp3`、`flac`、`pcm`、`aac`、`opus`，默认 `wav` |
| `speed` | number | 否 | 非流式输出时调整播放速度，范围 `0.25` 到 `4.0` |

#### vLLM-Omni 扩展字段

| 参数 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `task_type` | string | 否 | `CustomVoice`、`Base`、`VoiceDesign`；传 `ref_audio/ref_text` 时默认 `Base` |
| `language` | string | 否 | 兼容字段，当前仅记录日志 |
| `instructions` | string | 否 | 兼容字段，当前仅记录日志 |
| `max_new_tokens` | integer | 否 | 兼容字段，当前仅记录日志 |
| `initial_codec_chunk_frames` | integer | 否 | 兼容字段，当前仅记录日志 |
| `stream` | boolean | 否 | `true` 时流式返回 PCM，要求 `response_format=pcm` |
| `ref_audio` | string | Base 必需 | 参考音频，支持 URL、data URL、`file://`、绝对路径、相对项目根目录路径 |
| `ref_text` | string | Base 必需 | 参考音频对应文本，不需要传 CosyVoice3 前缀 |
| `x_vector_only_mode` | boolean | 否 | 当前不支持 `true` |

当前实现限制：

- `VoiceDesign` 暂不支持，会返回 400。
- `x_vector_only_mode=true` 暂不支持，会返回 400。
- `stream=true` 时必须设置 `response_format=pcm`。
- `stream=true` 时暂不支持 `speed` 调整。

## 参考音频要求

- 建议 WAV 格式。
- 建议 5 到 15 秒。
- 建议单声道，16kHz 或 24kHz。
- `ref_text` 必须与参考音频内容一致。
- `ref_text` 不需要包含 `You are a helpful assistant.<|endofprompt|>`，服务端会自动补齐。

参考音频路径说明：

- `asset/demo.wav` 会解析为容器内 `/app/asset/demo.wav`。
- `/app/asset/demo.wav` 会作为绝对路径使用。
- `file:///app/asset/demo.wav` 会解析为本地文件。
- `https://.../demo.wav` 会下载到临时文件后使用。
- `data:audio/wav;base64,...` 会写入临时文件后使用。

参考音频不存在时，接口返回：

```json
{
  "error": {
    "message": "参考音频不存在，请检查 ref_audio 路径：asset/missing.wav",
    "type": "invalid_request_error",
    "param": "ref_audio",
    "code": "reference_audio_not_found"
  },
  "resolved_path": "/app/asset/missing.wav"
}
```

## Docker Compose 常用命令

```bash
cd docker

# 构建镜像
docker compose build

# 前台启动
docker compose up

# 后台启动
docker compose up -d

# 查看日志
docker compose logs -f cosyvoice3

# 重启服务
docker compose restart cosyvoice3

# 停止并删除容器
docker compose down

# 进入容器
docker compose exec cosyvoice3 bash
```

## 环境变量

主要配置在 [docker/.env](docker/.env)：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST_PORT` | `10096` | 宿主机端口 |
| `SERVER_PORT` | `10096` | 容器内服务端口 |
| `GPU_DEVICE_ID` | `1` | 使用哪张 GPU；单卡通常改为 `0` |
| `MODEL_DIR` | `/app/models/Fun-CosyVoice3-0.5B` | 容器内模型目录 |
| `USE_VLLM` | `true` | 是否启用 vLLM |
| `USE_FP16` | `false` | 是否启用 FP16 |
| `OUTPUT_SAMPLE_RATE` | `24000` | 输出采样率，可设为 `16000` 或 `24000` |
| `SHM_SIZE` | `8g` | 容器共享内存大小 |

## 故障排查

### 容器无法看到 GPU

先确认宿主机可用：

```bash
nvidia-smi
```

再确认 Docker 可用：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

如果 Docker 命令失败，需要安装或修复 NVIDIA Container Toolkit。

### 启动提示模型不存在

确认宿主机模型目录：

```bash
ls models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
```

如果不存在，运行：

```bash
cd docker
docker compose run --rm --entrypoint bash cosyvoice3 -lc "cd /app && python download_model.py"
```

### `ref_audio` 找不到

容器内工作目录是 `/app`。相对路径会相对 `/app` 解析。

例如：

```json
{
  "ref_audio": "asset/longyingwan_woman.wav"
}
```

对应容器内文件：

```text
/app/asset/longyingwan_woman.wav
```

### 输出 PCM 无法播放

PCM 是裸音频，没有 WAV 头。需要指定采样率、位深和声道：

```bash
ffplay -f s16le -ar 24000 -ac 1 test.pcm
```

### 生成音色不像参考音频

检查以下项：

- `ref_text` 是否与参考音频逐字一致。
- 参考音频是否清晰、无噪声、无背景音乐。
- 参考音频时长是否在 5 到 15 秒左右。
- 合成文本不要过短，过短文本可能削弱音色表现。

## 本地开发

Docker 是推荐部署方式。本地调试仍可直接运行：

```bash
python cosyvoice_server.py \
  --host 0.0.0.0 \
  --port 10096 \
  --model_dir models/Fun-CosyVoice3-0.5B \
  --device cuda \
  --use_vllm
```
