#!/bin/bash
# Fun-CosyVoice3-0.5B-2512 启动脚本
# 作者：凌封
# 来源：https://aibook.ren (AI全书)

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CONDA_ENV_NAME="cosyvoice"
MODELS_DIR="$SCRIPT_DIR/models"
MODEL_PATH="$MODELS_DIR/Fun-CosyVoice3-0.5B"

# 0. 初始化并激活 Conda 环境
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Error: Conda 环境 '$CONDA_ENV_NAME' 不存在"
    echo "请先运行: ./install.sh"
    exit 1
fi

echo "激活环境: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME"

# 设置 cuDNN 库路径 (ONNX Runtime GPU 加速)
CUDNN_LIB=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)/lib
if [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
    echo "✓ 已启用 cuDNN GPU 加速"
fi

# 1. 检查模型是否存在
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/cosyvoice3.yaml" ]; then
    echo "Error: 模型未找到: $MODEL_PATH"
    echo "请先运行: python download_model.py"
    exit 1
fi

# 2. 启动服务
echo "=== 启动 CosyVoice TTS 服务 ==="
echo "模型路径: $MODEL_PATH"
echo "端口: 10096"
echo ""

cd "$SCRIPT_DIR"

# 启动服务
# Tips:
# 1. 如需开启 vLLM 加速 (需先安装 vllm)，运行: ./start_server.sh --use_vllm
# 2. 如需开启 FP16 (节省显存)，运行: ./start_server.sh --fp16
# 3. 输出采样率默认 16kHz (兼容小智平台)，如需原生 24kHz: ./start_server.sh --output_sample_rate 24000

python cosyvoice_server.py \
    --host 0.0.0.0 \
    --port 10096 \
    --model_dir "$MODEL_PATH" \
    --device cuda "$@"

