cd /c22940/zy/code/VLM-R1/src/open-r1-multimodal

export PYTHONWARNINGS="ignore:None of the inputs have requires_grad=True. Gradients will be None:UserWarning"
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=2,3,4,5

RUN_NAME="Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click-MajorityVoting-Temp0.8-wp4"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec_click.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /c22940/zy/model/Qwen2.5-VL-7B-Instruct \
    --dataset_name data_config/rec_with_screenspot_click_wp4.yaml \
    --image_root /c22940/zy/code/VLM-R1/otherdata/ScreenSpot-v2 \
    --max_prompt_length 1024 \
    --num_generations 16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 10 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --temperature 0.8 \
    --reward_funcs majority_click click_format 
