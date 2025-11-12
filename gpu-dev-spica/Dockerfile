# =========================
# GPU Dev Environment (CUDA 11.6 / cuDNN8 / Ubuntu 20.04)
# =========================
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# 基本ツールと Python ビルド依存
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    git curl wget ca-certificates build-essential \
    openssh-client vim less tzdata locales \
    libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
    libreadline-dev libsqlite3-dev libgdbm-dev libgdbm-compat-dev \
    libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.10.14
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd / && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

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
RUN /usr/local/bin/python3.10 -m venv ~/.venv && \
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


