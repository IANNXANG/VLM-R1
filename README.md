# VLM-R1

![Image](https://github.com/user-attachments/assets/e86a3ff2-a9c6-4548-8200-6c3c382d60e6)

![Image](https://github.com/user-attachments/assets/b3512920-ef30-4d6d-9bfe-c64e4570a067)

![image](https://github.com/user-attachments/assets/42b79f44-1c09-4c22-bad9-17ec2a0a1d10)

![image](https://github.com/user-attachments/assets/f5191b1e-dde2-42b7-9ec9-10f7f6213c12)


## Setup

```bash
bash setup.sh
```


## Training

### Counting VQA
> 1. Download the training data [🤗 Train Dataset: CLEVR-70k](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train)

> 2. ```bash src/open-r1-multimodal/run_grpo.sh```

```bash
cd src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir <OUTPUT_DIR> \
    --model_name_or_path <PATH-TO-Qwen2-VL-2B-Instruct> \
    --dataset_name <PATH-TO-DATASET-In-Repo> \ 
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true

```

> [!NOTE] 
> 1. To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. See the [reproduction report](https://github.com/Deep-Agent/R1-V/issues/4#issuecomment-2633348354) here. We realize it is important for effiency and are working on solving it with the community.
> 2. To reduce GPU cost with deepspeed and train larger model, please refer to this solution https://github.com/Deep-Agent/R1-V/issues/18.

### Referring Expression Comprehension (REC)

> 1. Download the [COCO Train2014 image](http://images.cocodataset.org/zips/train2014.zip).

> 2. Download the [RefCOCO/+/g Annotation files](https://huggingface.co/datasets/SZhanZ/mmc4_jsonl/resolve/main/rec_jsons_processed.zip)

> 3. Write the path of the annotation files in the `data_script/rec.yaml` file.
```bash
datasets:
    - json_path: /path/to/refcoco_train.json
    - json_path: /path/to/refcocop_train.json
    - json_path: /path/to/refcocog_train.json
```

> 4. ```bash src/open-r1-multimodal/run_grpo_rec.sh```

```bash
cd src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2-VL-2B-GRPO-REC"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_rec.py \
    --deepspeed ./local_scripts/zero3.json \
    --output_dir <OUTPUT_DIR>/$RUN_NAME \
    --model_name_or_path <PATH-TO-Model> \
    --dataset_name ./data_script/rec.yaml \
    --image_root <PATH-TO-Image-Root> \
    --max_prompt_length 1024 \
    --num_generations 8 \ # Could be reduced for faster training
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
```

## Evaluation

![image](https://github.com/user-attachments/assets/70a3cb1a-4588-48aa-9469-011fbd776be3)

We provide the example script to evaluate OOD counting performance on a subset of SuperCLEVR within 1 minute. You can also modify the script and dataset to test on your own dataset.



```bash
cd ./src/eval
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip
unzip images.zip

# change the model path in the script
python test_qwen2vl_counting_superclevr.py 

```




