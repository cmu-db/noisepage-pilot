# syntax=docker/dockerfile:1
# NoisePage officially supports Ubuntu 20.04.
FROM ubuntu:20.04

USER root

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
    # Postgres requirements.
    bison \
    build-essential \
    flex \
    libreadline-dev \
    libssl-dev  \
    libxml2-dev  \
    libxml2-utils \ 
    libxslt-dev \ 
    xsltproc  \
    zlib1g-dev \
    # BenchBase requirements.
    openjdk-17-jdk \
    unzip \ 
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade psutil

# Create a non-sudo user and switch to them.
# This is because PostgreSQL binaries don't like being
# run as root, e.g., initdb.
RUN useradd --create-home --shell /bin/bash --home-dir /home/terrier --password "$(openssl passwd -1 terrier)" -G sudo terrier 
USER terrier
ENV HOME=/home/terrier
WORKDIR ${HOME}
ENV PATH="${HOME}/.local/bin/:${PATH}"


ARG DEBIAN_FRONTEND=noninteractive


# Get noisepage-pilot repository.
RUN git clone https://github.com/garrisonhess/noisepage-pilot/
WORKDIR ${HOME}/noisepage-pilot

# Install required python modules for package noisepage-pilot installation.
RUN python3 -m pip install --user --upgrade pip setuptools wheel build

# Install noisepage-pilot requirements into ~/.local.
RUN python3 -m pip install --user --upgrade -r requirements.txt

# Build behavior package and install all dependencies (invokes setup.py).
RUN python3 -m build .

# RUN ["/bin/bash"]

CMD /bin/bash
ENTRYPOINT /bin/bash


  

# COPY ./docker-entrypoint.sh /
# ENTRYPOINT ["/docker-entrypoint.sh"]
