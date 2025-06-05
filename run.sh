#!/usr/bin/env bash

IMAGE="fenicsx:latest"
REQ="requirements.txt"
CONTAINER_REQ_PATH="/work/requirements.txt"

# 1. If the image does not exist, build it
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found. Building..."
  docker build -t "$IMAGE" .
else
  echo "Image $IMAGE found."
fi

# 2. Checksum of the local requirements.txt
LOCAL_SUM=$(sha256sum "$REQ" | awk '{print $1}')

# 3. Checksum of the requirements.txt in the container
IMAGE_SUM=$(docker run --rm "$IMAGE" \
            sha256sum "$CONTAINER_REQ_PATH" | awk '{print $1}' )

# 4. Compare checksums, if they are different, rebuild the image
if [ "$LOCAL_SUM" != "$IMAGE_SUM" ]; then
  echo "requirements.txt has changed. Rebuilding image..."
  docker build -t "$IMAGE" .
else
  echo "requirements.txt has not changed. No need to rebuild."
fi

# 5. Run the container
docker run --rm -it \
  -v "$(pwd)":/work \
  -w /work \
  "$IMAGE" \
  bash