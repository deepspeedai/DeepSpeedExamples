#!/bin/bash

echo $*

SCRIPT_DIR=$(dirname $(realpath $0))
HOST_IP=$(hostname -i)
NUM_NODES=${NUM_NODES:-1}

# verify that NUM_NODES is a positive integer
if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_NODES must be a positive integer"
    exit 1
fi

# check if NUM_NODES ==1 or hostfile_n${NUM_NODES} exists
if [ ! -f hostfile_n${NUM_NODES} ] && [ "${NUM_NODES}" != "1" ]; then
    echo "Error: hostfile_n${NUM_NODES} does not exist"
    exit 1
fi

if [ "${NUM_NODES}" == "1" ]; then
    # avoid dependency on pdsh when possible
    cd ${SCRIPT_DIR}; bash ./run.sh --host-ip ${HOST_IP} $*
else
    ds_ssh -f hostfile_n${NUM_NODES} "cd ${SCRIPT_DIR}; NUM_NODES=${NUM_NODES} bash ./run.sh --host-ip ${HOST_IP} $*"
fi
