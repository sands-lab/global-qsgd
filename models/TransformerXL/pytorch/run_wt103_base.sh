#!/bin/bash

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export OMP_NUM_THREADS=1

if [[ "$1" == 'train' ]]; then
    echo 'Run training...'
    python3 -m torch.distributed.launch --nproc_per_node="$2" train.py \
    --config_file wt103_base.yaml --powersgd_rank $3 --work_dir "$4" --eval_interval 500 --seed $5 --method $6 \
        "${@:7}"
elif [[ "$1" == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 -m torch.distributed.launch --nproc_per_node="$2" eval.py \
        --config_file wt103_base.yaml \
        "${@:3}"
else
    echo 'unknown argment 1'
fi