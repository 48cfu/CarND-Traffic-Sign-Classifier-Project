FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER @48cfu

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Make sure CUDNN is detected
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# Pick up some TF dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
	wget \
	apt-utils \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libicu-dev \
	libopenblas-dev \ 
	libgl1-mesa-dev \
	libgl1-mesa-glx \
	libosmesa6-dev \
        pkg-config \
	python \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
	git \
	tcl-dev \
	tk-dev \
	openjdk-8-jdk \
	gcc-4.8 \
	g++-4.8 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh tmp/Miniconda3-4.4.10-Linux-x86_64.sh
RUN bash tmp/Miniconda3-4.4.10-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/

COPY environment.yml .
RUN conda env create -f environment.yml

# Cleanup tarballs and downloaded package files
RUN conda clean -tp -y

RUN ln -s /usr/local/cuda/lib64/libcudnn.so.5 /usr/local/cuda/lib64/libcudnn.so

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Term 1 workdir
RUN mkdir /src
WORKDIR "/src"

# TensorBoard
EXPOSE 6006
# Jupyter
EXPOSE 8888
# Flask Server
EXPOSE 4567

## Two Birds, One Stone
# 1. sources conda environment
# 2. prevents the zombie container issue when started as pid 1

COPY run.sh /
RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]

CMD [ "/bin/bash" ]
