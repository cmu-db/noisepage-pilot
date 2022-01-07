#!/usr/bin/bash 

# get the repo
git clone https://github.com/cmu-db/noisepage-pilot && cd noisepage-pilot

# get venv because ubuntu 20.04 (server) doesn't come with it
sudo apt update && sudo apt install python3.8-venv

# install required python modules for package noisepage-pilot installation
python3 -m pip install --user --upgrade pip setuptools wheel build

# change this to append to ~/.bashrc or ~/.bash_profile? 
export PATH=${PATH}:${HOME}/.local/bin

# build behavior package and install all dependencies (invokes setup.py)
python3 -m build .

# install noisepage-pilot requirements into ~/.local
pip install -r requirements.txt

# run data generation
doit behavior --all
