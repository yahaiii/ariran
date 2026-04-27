FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Keep image setup lean; project dependencies are installed post-create via pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
