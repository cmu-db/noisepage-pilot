# NoisePage officially supports Ubuntu 20.04.
FROM ubuntu:20.04

# Suppress interactive dialog.
ARG DEBIAN_FRONTEND=noninteractive

# NoisePage Pilot requirements.
RUN apt update
RUN apt install -y sudo git python3 python3-pip python3.8-venv

# Postgres compilation requirements.
RUN apt install -y bison build-essential flex libreadline-dev libssl-dev libxml2-dev libxml2-utils libxslt-dev xsltproc zlib1g-dev

# Install BenchBase requirements.
RUN apt install -y openjdk-17-jdk unzip

# Create a non-sudo user and switch to them.
# This is because PostgreSQL binaries don't like being
# run as root, e.g., initdb.
RUN useradd --create-home --shell /bin/bash --home-dir /home/terrier --password "$(openssl passwd -1 terrier)" -G sudo terrier 
USER terrier
WORKDIR /home/terrier/
RUN echo "PATH=${PATH}:/home/terrier/.local/bin" >> /home/terrier/.bashrc

# Get noisepage-pilot repository.
RUN git clone --branch gh/improve-install-process https://github.com/garrisonhess/noisepage-pilot/
WORKDIR /home/terrier/noisepage-pilot

# Install required python modules for package noisepage-pilot installation.
RUN python3 -m pip install --user --upgrade pip setuptools wheel build

# Install noisepage-pilot requirements into ~/.local.
RUN python3 -m pip install --user --upgrade -r requirements.txt

# Build behavior package and install all dependencies (invokes setup.py).
RUN python3 -m build .

RUN ["/bin/bash"]
