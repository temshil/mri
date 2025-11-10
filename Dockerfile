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
	dc \
        zlib1g-dev \
        libssl-dev \
        python3 \
        python3-pip \
        curl \
        python3-dev \
	ffmpeg \
	libsm6 \
	libxext6 \
        libopenblas-dev \
        liblapack-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /temshil

# Build and install NiftyReg (latest)
RUN git clone https://github.com/KCL-BMEIS/niftyreg.git NiftyReg && \
    mkdir -p NiftyReg/build && cd NiftyReg/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/niftyreg .. && \
    make -j$(nproc) && make install

ENV LD_LIBRARY_PATH="/opt/niftyreg/lib"
ENV PATH="/opt/niftyreg/bin:${PATH}"

# Install FSL (latest, using official installer)
RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py -o fslinstaller.py && \
    python3 fslinstaller.py -d /usr/local/fsl -V latest -q && \
    rm fslinstaller.py

ENV FSLDIR=/usr/local/fsl
ENV PATH=${FSLDIR}/bin:$PATH
ENV LD_LIBRARY_PATH=${FSLDIR}/lib:${LD_LIBRARY_PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Download DSI Studio v2025.04.16 for Ubuntu 22.04
RUN wget https://github.com/frankyeh/DSI-Studio/releases/download/2024.06.12/dsi_studio_ubuntu2204.zip && \
    unzip dsi_studio_ubuntu2204.zip -d dsi_studio && \
    rm dsi_studio_ubuntu2204.zip

ENV PATH="/temshil/dsi_studio:$PATH"

COPY src/ src/
RUN chmod u+x src/convert2bids.sh
RUN chmod u+x src/batch_dwi.sh
RUN chmod u+x src/batch_fmri_part1.sh
RUN chmod u+x src/batch_fmri_part2.sh
COPY lib/ lib/