#!/bin/bash
# Fun-CosyVoice3-0.5B-2512 环境安装脚本 (独立部署版)
# 作者：凌封
# 来源：https://aibook.ren (AI全书)
#
# 本脚本会创建独立的 conda 环境 'cosyvoice'，适合单独开源使用

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CONDA_ENV_NAME="cosyvoice"

echo "=== 开始安装 Fun-CosyVoice3 环境 ==="
echo "工作目录: $SCRIPT_DIR"
echo "Conda 环境名: $CONDA_ENV_NAME"

# 抑制 pip 在 root 用户下运行的警告
export PIP_ROOT_USER_ACTION=ignore

# 1. 检查 Conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "Error: 未检测到 conda，请先安装 Miniconda 或 Anaconda"
    echo "安装指南: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 2. 初始化 conda (确保 conda activate 可用)
eval "$(conda shell.bash hook)"

# 3. 检查/创建 conda 环境
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "检测到已有环境 '$CONDA_ENV_NAME'，直接使用"
else
    echo "正在创建 Conda 环境: $CONDA_ENV_NAME (Python 3.10)..."
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

# 4. 激活环境
echo "激活环境: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME"

# 5. 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 6. 安装 PyTorch (CUDA 12.x)
echo "正在安装 PyTorch..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. 安装 CosyVoice 核心依赖
echo "正在安装 CosyVoice 依赖..."

# Linux 服务器使用 onnxruntime-gpu (需要从微软源安装)
pip install onnxruntime-gpu==1.18.0 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 安装其他依赖
pip install -r "$SCRIPT_DIR/requirements.txt" \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 7.1 (可选) 安装 vLLM 加速库
echo ""
read -p "是否安装 vLLM 加速库 (v0.9.0)? 推荐显存充足使用 [y/N]: " install_vllm
if [[ "$install_vllm" =~ ^[Yy]$ ]]; then
    echo "正在安装 vLLM v0.9.0..."
    # vLLM 需要 triton 等依赖，通常比较大
    pip install vllm==0.9.0 -i https://mirrors.aliyun.com/pypi/simple/
    echo "✓ vLLM 安装完成"
    
    # [重要] vLLM 可能会自动升级 numpy 到 2.x，导致 onnxruntime 报错
    # 这里强制降级回 numpy<2
    echo "正在检查 NumPy 版本兼容性..."
    pip install "numpy<2" -i https://mirrors.aliyun.com/pypi/simple/
else
    echo "跳过 vLLM 安装"
fi

# 8. 克隆 CosyVoice 官方源码 (如果不存在)
COSYVOICE_SRC="$SCRIPT_DIR/official"
if [ ! -d "$COSYVOICE_SRC" ]; then
    echo "正在克隆 CosyVoice 官方源码到 official 目录..."
    git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_SRC"
    cd "$COSYVOICE_SRC"
    git submodule update --init --recursive
    cd "$SCRIPT_DIR"
else
    echo "✓ CosyVoice 官方源码已存在: $COSYVOICE_SRC"
fi

# 9. 创建软链接
if [ ! -L "$SCRIPT_DIR/cosyvoice" ]; then
    ln -sf "$COSYVOICE_SRC/cosyvoice" "$SCRIPT_DIR/cosyvoice"
    echo "✓ 创建 cosyvoice 模块链接"
fi

if [ ! -L "$SCRIPT_DIR/third_party" ]; then
    ln -sf "$COSYVOICE_SRC/third_party" "$SCRIPT_DIR/third_party"
    echo "✓ 创建 third_party 模块链接"
fi

# 10. 创建必要目录
mkdir -p "$SCRIPT_DIR/models"
mkdir -p "$SCRIPT_DIR/asset"
mkdir -p "$SCRIPT_DIR/output"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "后续步骤："
echo "1. 激活环境: conda activate $CONDA_ENV_NAME"
echo "2. 下载模型: python download_model.py"
echo "3. 启动服务: ./start_server.sh"

# ==========================================
# 人工手动安装步骤说明 (仅供参考)
# ==========================================
# 如果您希望手动一步步执行安装，请参考以下命令：
#
# ==========================================
# 1. 进入部署目录
# ==========================================
# cd /data/cosyvoice
#
# ==========================================
# 2. 创建独立 Conda 环境
# ==========================================
# conda create -n cosyvoice python=3.10 -y
# conda activate cosyvoice
#
# ==========================================
# 3. 升级 pip
# ==========================================
# pip install --upgrade pip
#
# ==========================================
# 4. 安装 PyTorch (CUDA 12.x 版本)
# ==========================================
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# 验证 PyTorch 安装:
# python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
#
# ==========================================
# 5. 安装 onnxruntime-gpu (Linux 专用)
# ==========================================
# pip install onnxruntime-gpu==1.18.0 \
#     --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
#
# ==========================================
# 6. 安装 CosyVoice 核心依赖
# ==========================================
# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
#
# ==========================================
# 6.1 (可选) 安装 vLLM 加速库
# ==========================================
# pip install vllm==0.9.0 -i https://mirrors.aliyun.com/pypi/simple/
# [重要] vLLM 可能会自动升级 numpy 到 2.x，导致 onnxruntime 报错
# 这里强制降级回 numpy<2
# pip install "numpy<2" -i https://mirrors.aliyun.com/pypi/simple/
#
# ==========================================
# 7. 克隆 CosyVoice 官方源码
# ==========================================
# git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git official
# cd official && git submodule update --init --recursive && cd ..
#
# ==========================================
# 8. 创建软链接
# ==========================================
# ln -sf official/cosyvoice ./cosyvoice
# ln -sf official/third_party ./third_party
#
# 验证链接:
# ls -la cosyvoice third_party
#
# ==========================================
# 9. 创建必要目录
# ==========================================
# mkdir -p models asset output
#
# ==========================================
# 10. 下载模型
# ==========================================
# python download_model.py
#
# 或者手动使用 modelscope 下载:
# python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='models/Fun-CosyVoice3-0.5B')"
#
# 或者使用 git lfs 下载:
# git lfs install
# git clone https://www.modelscope.cn/FunAudioLLM/Fun-CosyVoice3-0.5B-2512.git models/Fun-CosyVoice3-0.5B
#
# ==========================================
# 11. 测试本地推理 (可选)
# ==========================================
# python test_inference.py
#
# ==========================================
# 12. 启动服务
# ==========================================
# python cosyvoice_server.py --port 10096 --model_dir models/Fun-CosyVoice3-0.5B --device cuda
#
# 或者使用启动脚本:
# ./start_server.sh
#
# ==========================================
# 11. 验证服务
# ==========================================
# 健康检查:
# curl http://localhost:10096/health
#
# 测试 TTS (需要新开终端):
# python test_client.py --text "你好，我是小智"
#
# 或使用 curl:
# curl -X POST http://localhost:10096/v1/audio/speech \
#   -H "Content-Type: application/json" \
#   -d '{"input":"你好","reference_aduio":"asset/longyingwan_woman.wav","reference_text":"我们将为全球城市的可持续发展贡献力量。"}' \
#   -o test.wav
#
# ==========================================
