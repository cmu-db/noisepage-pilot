# NoisePage-Pilot Docker Information

The `docker` folder is for all files that relate to development or CI docker images and containers.

## Tips

- Run the below commands from the project root.
- You must rebuild the Docker image every time there are changes in your source tree.

## Helpful Commands

To build a local development image you can use:

```docker build -f ./docker/Dockerfile --tag noisepage-pilot_dev .```

To build a container using the current `main` branch reference on GitHub, you can use:

```docker build -f ./docker/Dockerfile --tag noisepage-pilot_main https://github.com/cmu-db/noisepage-pilot.git#main```

To run the Docker container you can use:

```docker run -it --rm --privileged --cap-add="ALL" noisepage-pilot_dev```

To delete old images, you can use:

```docker system prune --all```
