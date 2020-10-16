#!/bin/sh

SCRIPT_DIR=$(dirname $(readlink -f ${0}))

SRC_DIR=${SCRIPT_DIR}/githooks
DEST_DIR=${SCRIPT_DIR}/../.git/hooks

echo "Installing githooks..."
if [ -d ${DEST_DIR} ]; then
  cp ${SRC_DIR}/* ${DEST_DIR}/
fi
