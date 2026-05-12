FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=1
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONPATH=/app:/app/third_party/Matcha-TTS

WORKDIR /app

# =========================================================
# 1. 系统依赖：apt 使用清华 Ubuntu 镜像
# =========================================================
RUN sed -i 's@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    build-essential \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    unzip \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 2. 安装 Miniforge：使用清华镜像
# =========================================================
RUN wget -q "https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh && \
    conda clean -afy

# =========================================================
# 3. 配置 conda：只使用清华 conda-forge 镜像
# =========================================================
RUN conda config --system --remove-key channels || true && \
    conda config --system --set show_channel_urls yes && \
    conda config --system --set channel_priority strict && \
    conda config --system --add channels "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge" && \
    conda create -n cosyvoice python=3.10 -y \
      --override-channels \
      -c "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge" && \
    conda clean -afy

SHELL ["conda", "run", "-n", "cosyvoice", "/bin/bash", "-c"]

# =========================================================
# 4. 复制项目代码
# =========================================================
COPY . /app

# =========================================================
# 5. pip 国内源配置
# =========================================================
RUN mkdir -p /root/.pip && \
    printf "[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\ntrusted-host = mirrors.aliyun.com\ntimeout = 120\n" > /root/.pip/pip.conf && \
    python -m pip install --upgrade pip setuptools wheel

# =========================================================
# 6. 安装 PyTorch CUDA 12.1
# =========================================================
RUN pip install torch torchaudio \
    --index-url "https://mirrors.nju.edu.cn/pytorch/whl/cu121"

# =========================================================
# 7. 安装 onnxruntime-gpu
# =========================================================
RUN pip install onnxruntime-gpu==1.18.0 \
    --extra-index-url "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

# =========================================================
# 8. 安装项目 requirements.txt
# =========================================================
RUN pip install -r /app/requirements.txt \
    -i "https://mirrors.aliyun.com/pypi/simple/" \
    --trusted-host="mirrors.aliyun.com"

# =========================================================
# 9. 固定安装 vLLM v0.9.0
# =========================================================
RUN pip install vllm==0.9.0 \
      -i "https://mirrors.aliyun.com/pypi/simple/" \
      --trusted-host="mirrors.aliyun.com" && \
    pip install "numpy<2" \
      -i "https://mirrors.aliyun.com/pypi/simple/" \
      --trusted-host="mirrors.aliyun.com"

# =========================================================
# 10. 固定克隆 CosyVoice official 源码，并执行 submodule
# =========================================================
RUN git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/" && \
    echo "克隆 CosyVoice official 源码..." && \
    git clone --depth 1 "https://gh-proxy.com/https://github.com/FunAudioLLM/CosyVoice.git" /app/official && \
    cd /app/official && \
    git submodule update --init --recursive

# =========================================================
# 11. 固定补齐 Matcha-TTS，创建软链接和必要目录
# =========================================================
RUN if [ ! -d "/app/official/cosyvoice" ]; then \
      echo "Error: /app/official/cosyvoice 不存在，CosyVoice official 源码不完整"; \
      exit 1; \
    fi && \
    mkdir -p /app/official/third_party && \
    if [ ! -d "/app/official/third_party/Matcha-TTS" ]; then \
      echo "补齐 third_party/Matcha-TTS..." && \
      git clone --depth 1 \
        "https://gh-proxy.com/https://github.com/shivammehta25/Matcha-TTS.git" \
        /app/official/third_party/Matcha-TTS; \
    fi && \
    ln -sfn /app/official/cosyvoice /app/cosyvoice && \
    ln -sfn /app/official/third_party /app/third_party && \
    mkdir -p /app/models /app/asset /app/output /root/.cache/modelscope

# =========================================================
# 12. 验证关键依赖
# =========================================================
RUN python - <<'PY'
import torch
from matcha.models.components.flow_matching import BASECFM

print("PyTorch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Matcha-TTS OK:", BASECFM)
PY

# =========================================================
# 13. 入口脚本
# =========================================================
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 10096

ENTRYPOINT ["/docker-entrypoint.sh"]