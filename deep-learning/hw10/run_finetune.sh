#!/bin/bash

DIR_NAME=$(cd "$(dirname "$0")" && pwd)

echo Executing GPT2 Finetune
python "${DIR_NAME}/finetune.py" --model_type gpt2 --epochs 1 --output_dir "${DIR_NAME}/gpt2-samsum" | tee "${DIR_NAME}/gpt2.log"

echo Executing T5 Finetune
python "${DIR_NAME}/finetune.py" --model_type t5 --epochs 1 --output_dir "${DIR_NAME}/t5-samsum" | tee "${DIR_NAME}/t5.log"
