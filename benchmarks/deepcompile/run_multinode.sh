#!/bin/bash

echo $*

SCRIPT_DIR=$(dirname $(realpath $0))
HOST_IP=$(hostname -i)
NUM_NODES=${NUM_NODES:-$(wc -l < /job/hostfile)}

if [ "${NUM_NODES}" == "1" ]; then
    # avoid dependency on pdsh when possible
    cd ${SCRIPT_DIR}; bash ./run.sh --host-ip ${HOST_IP} $*
else
    ds_ssh -f hostfile_n${NUM_NODES} "cd ${SCRIPT_DIR}; bash ./run.sh --host-ip ${HOST_IP} $*"
fi
