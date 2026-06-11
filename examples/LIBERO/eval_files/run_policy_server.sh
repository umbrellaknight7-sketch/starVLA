#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/home/xurui/starVLA
LIBERO_HOME=/home/xurui/LIBERO
STARVLA_PYTHON=/home/xurui/miniconda3/envs/starVLA/bin/python
LIBERO_PYTHON=/home/xurui/miniconda3/envs/libero/bin/python

# === Checkpoint ===
CKPT=/data/checkpoints_rui/starVLA_results/1229_libero3vl4B_qwen3gr00t_state/checkpoints/steps_25000_pytorch_model.pt

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=0
port=6694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
