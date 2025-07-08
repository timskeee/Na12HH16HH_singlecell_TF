FROM python:3.10-slim

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential git curl cmake \
    libx11-dev libsm6 libxext6 libxrender-dev \
    && apt-get clean

# Upgrade pip (NetPyNE dev branch installs everything via pip)
RUN python3 -m pip install --upgrade pip

# Clone and install NetPyNE in editable mode
WORKDIR /opt
RUN git clone https://github.com/Neurosim-lab/netpyne.git && \
    cd netpyne && \
    git checkout development && \
    pip install -e .

# Set working directory for the user
WORKDIR /SY

# Default to bash shell
CMD ["/bin/bash"]
