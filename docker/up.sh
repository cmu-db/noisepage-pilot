#!/bin/bash

docker build -f ./docker/Dockerfile --tag noisepage-pilot_dev .
docker run -it --rm --cap-add="ALL" noisepage-pilot_dev
