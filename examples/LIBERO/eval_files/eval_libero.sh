#!/bin/bash
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/home/xurui/starVLA

cd ${STARVLA_DIR}
# === Checkpoint ===
CKPT=/data/checkpoints_rui/starVLA_results/1229_libero3vl4B_qwen3gr00t/final_model/pytorch_model.pt

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/home/xurui/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/xurui/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host="127.0.0.1"
base_port=6694
unnorm_key="franka"
your_ckpt=${CKPT}

# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# model_root: playground/Checkpoints/<run_id>
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=libero_goal
num_trials_per_task=20
video_out_path="/home/xurui/starVLA/results/${task_suite_name}/${folder_name}"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
