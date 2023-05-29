#!/bin/bash

# default name for the container
NAME=does_clip_know_my_face
WANDB_KEY=""

help() {
  # display the help text
  echo "This script builds a docker image using the Dockerfile."
  echo 
  echo "Usage: docker_build.sh [OPTION...]"
  echo 
  echo "options:"
  echo "-n, --name      Specify a name for the Docker image. (Default: ${NAME})"
  echo "-w, --wandb     Specify your Weights and Biases API key to use WandB within the Docker container."
  echo "-h, --help      Prints help."
}

POSITIONAL=()
while [ $# -gt 0 ]; do
  key="$1"

  case $key in
    -n|--name)
      NAME="$2"
      shift # passed argument
      shift # passed value
      ;;
    -w|--wandb)
      WANDB_KEY="$2"
      shift
      shift
      ;;
    -h|--help)
      help
      exit 0
      ;;
    *)
      echo "Argument '$key' unknown"
      exit 1
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

echo "Image name: ${NAME}"
if [ -n "${WANDB_KEY}" ] ; then
  echo "WandB API key: ${WANDB_KEY}"
fi

docker build -t "${NAME}" --build-arg WANDB_KEY="${WANDB_KEY}" .