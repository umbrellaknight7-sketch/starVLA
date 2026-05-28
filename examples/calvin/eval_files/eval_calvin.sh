#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH} # let Calvin client find websocket tools from main repo
export calvin_python=/path/to/your/conda/envs/calvin/bin/python

host="127.0.0.1"
base_port=5694
your_ckpt=results/Checkpoints/0118_starvla_qwengr00t_calvin_task_D_D/checkpoints/steps_30000_pytorch_model.pt
dataset_path=/path/to/calvin/task_D_D/
calvin_config_path=/path/to/calvin/calvin_models/conf
eval_sequences_path=examples/calvin/eval_files/eval_sequences.json

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

${calvin_python} ./examples/calvin/eval_files/eval_calvin.py \
    --args.host "$host" \
    --args.port $base_port \
    --args.dataset-path ${dataset_path} \
    --args.calvin-config-path ${calvin_config_path} \
    --args.eval-sequences-path ${eval_sequences_path} \
    --args.eval-log-dir ${LOG_DIR} \
    --args.num-sequences 1000
