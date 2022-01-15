# syntax=docker/dockerfile:1
# NoisePage officially supports Ubuntu 20.04.
FROM ubuntu:20.04

# Suppress interactive dialog.
ARG DEBIAN_FRONTEND=noninteractive

# NoisePage Pilot requirements.
RUN set -eux; \
    apt-get update \
    && apt-get install -y \
    # Behavior modeling requirements.
    sudo \
    git \
    python3 \
    python3-pip \
    python3.8-venv \
    graphviz \ 
    # Postgres requirements.
    bison \
    build-essential \
    flex \
    libreadline-dev \
    libssl-dev \
    libxml2-dev \
    libxml2-utils \
    libxslt-dev \
    xsltproc \
    zlib1g-dev \
    # BenchBase requirements.
    openjdk-17-jdk \
    unzip \
    # TScout (BCC) requirements.
    bpfcc-tools \
    linux-headers-$(uname -r) \
    && rm -rf /var/lib/apt/lists/*

# TScout also requires psutil (on the root user).
RUN python3 -m pip install --upgrade psutil

RUN useradd --create-home --shell /bin/bash --home-dir /home/terrier \
    --password "$(openssl passwd -1 terrier)" -G sudo terrier  \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER terrier
ENV HOME=/home/terrier
WORKDIR $HOME

# Add pip package installation directory to PATH.
ENV PATH="${HOME}/.local/bin/:${PATH}"

# Copy the contents of the build context (working directory).
COPY --chown=terrier:terrier . ${HOME}/noisepage-pilot

# Set the pilot directory to current working directory.
WORKDIR ${HOME}/noisepage-pilot

# Install required Python packages and build the pilot.
RUN python3 -m pip install --user --upgrade -r requirements.txt \ 
    && doit noisepage_build benchbase_build

# Provide shell entrypoint.
ENTRYPOINT /bin/bash
