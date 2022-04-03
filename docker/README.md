# NoisePage-Pilot Docker Information

The `docker` folder is for all files that relate to development or CI docker images and containers.

A successful build-and-run will put you in the shell in the running container.  At this point, you should be able to run `doit` tasks or perform other general development tasks.

## Tips

- Run the below commands from the project root.
- You must rebuild the Docker image every time there are changes in your source tree.

## Helpful commands

To build a local development image you can use:

```docker build -f ./docker/Dockerfile --tag noisepage-pilot_dev .```

To build a container using the current `main` branch reference on GitHub, you can use:

```docker build -f ./docker/Dockerfile --tag noisepage-pilot_main https://github.com/cmu-db/noisepage-pilot.git#main```

To run the Docker container you can use:

```docker run -it --rm --cap-add="ALL" noisepage-pilot_dev```

To delete old images, you can use:

```docker system prune --all```

## Docker gotchas

Anything that takes you over 1 day to figure out should be documented here.
