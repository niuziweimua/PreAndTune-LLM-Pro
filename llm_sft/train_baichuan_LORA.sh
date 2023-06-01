# pip3 install -r requirements.txt

BASE_MODEL_PATH=/home/reacubeth/models/baichuan-2-7b-base-pretrain
EPOCH=10

TITLE=baichuan-2-7b-base-pretrain-lr1e5epoch5data12length512-LORA
DATA=data

OUTPUT_DIR=result

echo ===== current OUTPUT_DIR is $OUTPUT_DIR =====

torchrun --nproc_per_node=4 --master_port=2219 train_lora.py \
    --model_name_or_path $BASE_MODEL_PATH \
    --data_path $DATA \
    --bf16 T