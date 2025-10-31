# =========================
# GPU Dev Environment (CUDA 12.4 / cuDNN9 / Ubuntu 22.04)
# =========================
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

# 基本ツールと Python
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    git curl ca-certificates build-essential \
    openssh-client vim less tzdata locales && \
    rm -rf /var/lib/apt/lists/*

# ロケール設定
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ユーザ作成
ARG USERNAME=dev
ARG UID=1000
RUN useradd -m -u ${UID} -s /bin/bash ${USERNAME} && \
    mkdir -p /workspace && chown ${USERNAME}:${USERNAME} /workspace
WORKDIR /workspace
USER ${USERNAME}

# Python 仮想環境 + よく使うライブラリ
RUN python3 -m venv ~/.venv && \
    ~/.venv/bin/pip install --upgrade pip wheel setuptools && \
    ~/.venv/bin/pip install \
      ipython jupyter jupyterlab \
      numpy scipy pandas matplotlib \
      scikit-learn tqdm rich

# ---- PyTorch (CUDA 12.4) を使う場合 ----
# RUN ~/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# PATH 設定
ENV PATH="/home/${USERNAME}/.venv/bin:${PATH}"

EXPOSE 8888
CMD ["bash", "-lc", "python -c 'import subprocess; subprocess.run([\"nvidia-smi\"])' && bash"]

