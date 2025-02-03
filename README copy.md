# R1-V: Reinforce Super Generalization Ability in Vision Langauge Models with Less Than $3

*Contributors: Liang Chen · Lei Li · Haozhe Zhao · Yifan Song*

1. We firstly reveal that **Reinforcement Learning with Verifiable Rewards (RLVR)** outperforms chain-of-thought supervised fine-tuning (CoT-SFT) in both **effectiveness and out-of-distribution (OOD) robustness** for vision language models.

2. In our experiment, we **incentivize** VLMs to learn **generalizable** visual counting abilities, rather than overfitting to the training set.

3. The 2B model outperforms the 72B model in OOD tests within just **100** training steps.

4. The training was conducted on 8 A100 GPUs for **30 minutes, costing $2.62**.

5. Codes, models, datasets, more details and **all open-source** resources will be shared (within CNY holidays).

Latest: We upload the codebase.

---



![image](./images/ood.png)

![image](./images/super_ood.png)

![image](./images/training.png)

---

## Training

```bash
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.

# follow open-r1-multimodal to install the packages.

export DEBUG_MODE="true"
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



## Citation

```bib
@misc{R1-V,
author       = {Liang Chen and Lei Li and Haozhe Zhao and Yifan Song},
title        = {R1-V: Reinforce Super Generalization Ability in Vision Langauge Models with Less Than $3},
howpublished = {https://github.com/Deep-Agent/R1-V},
note         = {Accessed: 2025-02-02},
year         = {2025}
}

```


