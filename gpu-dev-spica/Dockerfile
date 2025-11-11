# =========================
# GPU Dev Environment (CUDA 11.6 / cuDNN8 / Ubuntu 20.04)
# =========================
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# 基本ツールと Python
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    lsb-release \
    git curl ca-certificates build-essential \
    openssh-client vim less tzdata locales && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3.10-distutils && \
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

# Python 仮想環境 + よく使うライブラリ（Python 3.10）
RUN python3.10 -m ensurepip --upgrade && \
    python3.10 -m venv ~/.venv && \
    ~/.venv/bin/pip install --upgrade pip wheel setuptools && \
    ~/.venv/bin/pip install \
      ipython jupyter jupyterlab \
      numpy scipy pandas matplotlib \
      scikit-learn tqdm rich

# ---- PyTorch (CUDA 11.6) を使う場合 ----
# RUN ~/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio

# PATH 設定
ENV PATH="/home/${USERNAME}/.venv/bin:${PATH}"

EXPOSE 8888
CMD ["bash", "-lc", "python -c 'import subprocess; subprocess.run([\"nvidia-smi\"])' && bash"]


