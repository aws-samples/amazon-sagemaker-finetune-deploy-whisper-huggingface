FROM nvcr.io/nvidia/pytorch:23.05-py3

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV NVIDIA_VISIBLE_DEVICES="all"

#install ffmpeg and sudo
RUN apt update && \
        apt install -y software-properties-common && \
        add-apt-repository -y ppa:jonathonf/ffmpeg-4 && \
        apt install -y ffmpeg && \
        apt install -y sudo && \
        rm -rf /var/lib/apt/lists/*

#install git-lfs
COPY script.deb.sh script.deb.sh
RUN bash script.deb.sh && \
        apt install -y git-lfs -y && \
        git lfs install

#install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
        pip uninstall -y jupyterlab_widgets && \
        pip install jupyterlab_widgets