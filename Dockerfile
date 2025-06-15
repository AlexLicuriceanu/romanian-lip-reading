FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    nano \
    ffmpeg \
    libturbojpeg0-dev \
    git \
    libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda create -n venv python=3.10 -y && \
    conda install -n venv pip -y && \
    conda clean -afy

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate venv" >> ~/.bashrc

SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-c"]

RUN mkdir -p /opt/conda/envs/venv/etc/conda/activate.d && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))")' \
    > /opt/conda/envs/venv/etc/conda/activate.d/env_vars.sh

WORKDIR /app

COPY ./models ./models
COPY ./talknet ./talknet
COPY ./requirements.txt ./
COPY *.sh ./
COPY *.txt ./
COPY *.py ./
COPY *.md ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -U yt-dlp && \
    pip cache purge

CMD ["/bin/bash"]
