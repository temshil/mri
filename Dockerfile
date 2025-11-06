FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    wget \
    unzip \
    git \
    build-essential \
    cmake \
    zlib1g-dev \
    libssl-dev \
    libsm6 \
    libxext6 \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    dc \
    libglib2.0-0 \
    libxrender1 \
    libxtst6 \
    libgtk2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /temshil

# Python venv
ENV VIRTUAL_ENV=/opt/env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt

# Build and install NiftyReg (latest)
RUN git clone https://github.com/KCL-BMEIS/niftyreg.git NiftyReg && \
    mkdir -p NiftyReg/build && cd NiftyReg/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/niftyreg .. && \
    make -j$(nproc) && make install

ENV PATH="/opt/niftyreg/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/niftyreg/lib:$LD_LIBRARY_PATH"

# Install FSL (latest, using official installer)
RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py -o fslinstaller.py && \
    python3 fslinstaller.py -d /usr/local/fsl -V latest -q

ENV FSLDIR=/usr/local/fsl
ENV PATH=${FSLDIR}/bin:$PATH
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Download DSI Studio v2025.04.16 for Ubuntu 22.04
RUN wget https://github.com/frankyeh/DSI-Studio/releases/download/2025.04.16/dsi_studio_ubuntu2204.zip && \
    unzip dsi_studio_ubuntu2204.zip -d dsi_studio && \
    rm dsi_studio_ubuntu2204.zip

ENV PATH="/temshil/dsi_studio:$PATH"