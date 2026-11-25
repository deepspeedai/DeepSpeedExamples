# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Sourced on each node before anything that runs NCCL across nodes (Ray's raylets, so every
# actor inherits it; torchrun). NCCL reaches the other node over EFA through the aws-ofi-nccl
# plugin on the AMI. LD_LIBRARY_PATH holds only the plugin and libfabric: the AMI's CUDA
# libraries break torch's cuDNN (bench/setup_node.sh).
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1
export NCCL_SOCKET_IFNAME=^lo,docker
