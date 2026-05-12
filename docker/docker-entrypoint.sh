#!/usr/bin/env bash
set -e

CONDA_ENV_NAME="${CONDA_ENV_NAME:-cosyvoice}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

SCRIPT_DIR="/app"
cd "$SCRIPT_DIR"

MODELS_DIR="$SCRIPT_DIR/models"
MODEL_PATH="${MODEL_DIR:-$MODELS_DIR/Fun-CosyVoice3-0.5B}"

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-10096}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda}"

USE_VLLM="${USE_VLLM:-true}"
USE_FP16="${USE_FP16:-false}"
OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-24000}"

echo "=================================================="
echo "Fun-CosyVoice3-0.5B-2512 Docker 启动"
echo "=================================================="
echo "Conda env:          $CONDA_ENV_NAME"
echo "Python:             $(which python)"
echo "SCRIPT_DIR:         $SCRIPT_DIR"
echo "MODEL_PATH:         $MODEL_PATH"
echo "SERVER_HOST:        $SERVER_HOST"
echo "SERVER_PORT:        $SERVER_PORT"
echo "SERVER_DEVICE:      $SERVER_DEVICE"
echo "USE_VLLM:           $USE_VLLM"
echo "USE_FP16:           $USE_FP16"
echo "OUTPUT_SAMPLE_RATE: $OUTPUT_SAMPLE_RATE"
echo "=================================================="

# =========================================================
# 1. 检查 PyTorch / CUDA
# =========================================================
python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# =========================================================
# 2. 设置 cuDNN Python wheel 库路径
# =========================================================
CUDNN_LIB=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)/lib || true

if [ -d "$CUDNN_LIB" ]; then
  export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
  echo "✓ 已启用 cuDNN 库路径: $CUDNN_LIB"
else
  echo "未检测到 nvidia.cudnn Python 包路径，继续启动"
fi

# =========================================================
# 3. 检查 official / cosyvoice / third_party
# =========================================================
if [ ! -d "$SCRIPT_DIR/official" ]; then
  echo "Error: /app/official 不存在"
  echo "请确认 Dockerfile 构建阶段已经克隆 CosyVoice official"
  exit 1
fi

if [ ! -d "$SCRIPT_DIR/official/cosyvoice" ]; then
  echo "Error: /app/official/cosyvoice 不存在"
  echo "CosyVoice official 源码不完整"
  exit 1
fi

if [ ! -d "$SCRIPT_DIR/official/third_party" ]; then
  echo "Error: /app/official/third_party 不存在"
  echo "CosyVoice third_party 不完整"
  exit 1
fi

ln -sfn "$SCRIPT_DIR/official/cosyvoice" "$SCRIPT_DIR/cosyvoice"
ln -sfn "$SCRIPT_DIR/official/third_party" "$SCRIPT_DIR/third_party"

# =========================================================
# 4. 设置 PYTHONPATH
# =========================================================
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/third_party/Matcha-TTS:${PYTHONPATH:-}"

if [ ! -d "$SCRIPT_DIR/third_party/Matcha-TTS/matcha" ]; then
  echo "Error: Matcha-TTS 不存在: $SCRIPT_DIR/third_party/Matcha-TTS/matcha"
  echo "请确认 Dockerfile 构建阶段已经补齐 Matcha-TTS"
  exit 1
fi

# =========================================================
# 5. 检查模型是否存在
# =========================================================
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/cosyvoice3.yaml" ]; then
  echo "Error: 模型不存在或不完整: $MODEL_PATH"
  echo ""
  echo "请先进入容器运行模型下载命令："
  echo ""
  echo "  cd /app"
  echo "  python download_model.py"
  echo ""
  echo "或者在宿主机提前下载模型到："
  echo ""
  echo "  ./models/Fun-CosyVoice3-0.5B"
  echo ""
  echo "然后通过 docker-compose volume 挂载到："
  echo ""
  echo "  /app/models/Fun-CosyVoice3-0.5B"
  echo ""
  exit 1
fi

# =========================================================
# 6. 检查 ttsfrd 状态，只检查，不安装
# =========================================================
python - <<'PY'
try:
    import ttsfrd
    print("✓ ttsfrd 已安装")
except Exception as e:
    print(f"⚠ ttsfrd 未安装或不可用: {e}")
    print("  如需 ttsfrd，请先运行 python download_model.py 下载 CosyVoice-ttsfrd，然后按项目提示手动安装")
PY

# =========================================================
# 7. 拼接 cosyvoice_server.py 启动参数
# =========================================================
CMD_ARGS=(
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --model_dir "$MODEL_PATH"
  --device "$SERVER_DEVICE"
  --output_sample_rate "$OUTPUT_SAMPLE_RATE"
)

if [ "$USE_VLLM" = "true" ]; then
  CMD_ARGS+=(--use_vllm)
fi

if [ "$USE_FP16" = "true" ]; then
  CMD_ARGS+=(--fp16)
fi

echo "=================================================="
echo "启动 CosyVoice TTS 服务"
echo "python cosyvoice_server.py ${CMD_ARGS[*]}"
echo "=================================================="

exec python cosyvoice_server.py "${CMD_ARGS[@]}"