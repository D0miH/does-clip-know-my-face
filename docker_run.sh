#!/bin/bash
IMAGE_NAME=does_clip_know_my_face
CONTAINER_NAME=does_clip_know_my_face
DEVICES=""
MOUNTING_FILE="mounts.docker"
SHM_SIZE="16G"
PORT_MAPPING=""

help() {
  # display the help text
  echo "This script runs a docker container."
  echo 
  echo "Usage: docker_run.sh [OPTION...]"
  echo 
  echo "options:"
  echo "-i, --image       Specify a name for the Docker image to run. (Default: ${NAME})"
  echo "-n, --name        Specify the name of the container that is going to be started. (Default: ${CONTAINER_NAME})"
  echo "-d, --devices     Specify the IDs of the GPUs to use (e.g. \"0,1\"). (Default: ${DEVICES})"
  echo "--shm-size        Shared memory size of the docker container. (Default: ${SHM_SIZE})"
  echo "-m, --mount_file  The mount file to use to mount symbolic links within the docker container. (Default: ${MOUNTING_FILE})"
  echo "-p, --port        Specify which ports of the container should be exposed."
  echo "-h, --help        Prints help."
}

POSITIONAL=()
while [ $# -gt 0 ]; do
  key="$1"

  case $key in
    -i|--image)
      IMAGE_NAME="$2"
      shift # passed argument
      shift # passed value
      ;;
    -n|--name)
      CONTAINER_NAME="$2"
      shift
      shift
      ;;
    -d|--devices)
      DEVICES="$2"
      shift
      shift
      ;;
    --shm-size)
      SHM_SIZE="$2"
      shift
      shift
      ;;
    -m|--mount_file)
      MOUNTING_FILE="$2"
      shift
      shift
      ;;
    -p|--port)
      PORT_MAPPING="$2"
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

DEVICE_COMMAND=""
if [[ $DEVICES =~ ^[0-9]+$ ]] ; then
  DEVICE=$(echo "$DEVICES" | tr -d '"')
  DEVICE_COMMAND=\""device=$DEVICE"\"
else
  DEVICE_COMMAND='all'
fi


ADDITIONAL_MOUNTING_COMMAND=""
if [ -n "${MOUNTING_FILE}" ] ; then
  while read -r line; do
    [[ "$line" =~ ^#.*$ ]] && continue
    ADDITIONAL_MOUNTING_COMMAND+=" -v \$(pwd)$line:/workspace$line"
  done < "$MOUNTING_FILE"
fi

PORT_MAPPING_CMD=""
if [ -n "${PORT_MAPPING}" ] ; then
  PORT_MAPPING_CMD="-p ${PORT_MAPPING} "
fi

echo "----------Running the following command:----------"
echo "docker run --rm --shm-size ${SHM_SIZE} --name ${CONTAINER_NAME} --gpus '${DEVICE_COMMAND}' -v \$(pwd):/workspace${ADDITIONAL_MOUNTING_COMMAND} ${PORT_MAPPING_CMD}-itd ${IMAGE_NAME} bash"
echo "--------------------------------------------------"
eval "docker run --rm --shm-size ${SHM_SIZE} --name ${CONTAINER_NAME} --gpus '${DEVICE_COMMAND}' -v \$(pwd):/workspace${ADDITIONAL_MOUNTING_COMMAND} ${PORT_MAPPING_CMD}-itd ${IMAGE_NAME} bash"