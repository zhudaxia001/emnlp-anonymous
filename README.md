# Curr-ReFT: Overcoming Training Bottlenecks in Small-scale Vision-Language Models via Curriculum Reinforcement Finetuning
- Updated pass@k evaluation code and usage instructions  --- 2025.05.21
- We updated our README.md to include code structure and usage instructions --- 2025.04.16
- We release our code. --- 2025.03.21
- We will update the data, code, and model weight, once the cor finish checking.

## Introduction
We deeply investigate the R1-like RL in VLM(MLLM), mainly for answering the following questions:
1. Can Rule-based Reinforcement Learning (RL) applied to multimodal mathematical data enhance general capabilities? 
2. Is it feasible to implement Rule-based RL in other multimodal tasks or computer vision (CV) tasks? If so, what specific improvements can be achieved?
3. For small-scale multimodal large models with limited baseline capabilities, can the aforementioned experiences be applicable? If yes, what optimization strategies should be adopted?
4. After the RL phase, is there a possibility of performance degradation in certain aspects of the model? How can the overall training process be refined to address this issue?



- The data is organized as follows:
 
```bash
-grpo_data
  -images
  -test
  -train
-SFT_data
  -reject
  -sft
  -images
```

* grpo_data: The "images" folder contains all image files. The "train" folder includes training and in-domain test JSONL files for all three stages (the test files inside are for in-domain scenarios), while the "test" folder contains only JSONL files for out-of-domain testing across the three tasks.
* SFT_data: The "reject" folder contains JSON files for rejection-based SFT, while the "sft" folder contains JSONL files for standard SFT (used for comparative experiments in the paper). All images are stored in the "images" folder.


## Requirement
1、We implete SFT training based on [MS-Swift](https://github.com/modelscope/ms-swift) 
for specifically we use ms-swift 3.2.0:
```bash
   conda create -n swift python=3.10
   conda activate swift
   pip install ms-swift==3.2.0
```
2、We perform GRPO based on [OpenR1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal/)
```bash
   conda create -n R1-V python=3.10
   conda activate R1-V
   pip install -r requirements_for_R1_V.txt
```
**Note**: Some packages in `requirements_for_R1_V.txt` require local installation. Please follow these steps:
1. Download the packages from the provided links in the requirements file
2. Install them using `pip install -e .` in their respective directories

## Training
```bash
Curr_REFT/
├── grpo_sft_data/
│   ├── grpo_data/
│   └── SFT_data/
└── train_code/
    ├── grpo/
    │   ├── Train_sh_files/
    │   ├── data_config/
    │   └── open-r1-multimodal/
    └── sft/
        ├── normal_sft/
        └── reject_sft/
```
We have placed the training scripts in the following directories:
```bash
-For Qwen2.5-VL-3B: Curr_REFT/train_code/grpo/Train_sh_files/Curriculum-based_RL_math_resize
-For Qwen2.5-VL-7B: Curr_REFT/train_code/grpo/Train_sh_files/7B_Curriculum-based_RL_math_resize
```
Hence, you only need to fine the right order and run .sh as the paper claims.

2、For three-stage curriculum reinforcement learning (GRPO), execute the following stages(using Qwen2.5-VL-3B as an example):
### Stage 1: Judge
First, activate the reinforcement learning environment:
```bash
 conda activate R1-V
```
The placeholders are included for you to replace with your actual paths.
```bash
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    /Curr_REFT/train_code/grpo/open-r1-multimodal/src/open_r1/Stage1_judge_math_resize.py \
    --deepspeed /mnt/tenant-home_speed/dhl/VLM-R1-main/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir /Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000 \ # output
    --model_name_or_path /mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-3B-Instruct \  # Path/to/Qwen2.5-VL-3B-Instruct
    --dataset_name /data_config/stage1_judge/3_tasks_4000.yaml \  # data_config
    --image_root /path/to/image_root \   #image_data_root
    --max_prompt_length 2048 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \  # $RUN_NAME
    --save_steps 100 \
    --save_only_model true
```
After replacing these placeholders with your actual paths, you can run the script for stage1:
```python
sh /Curr_REFT/train_code/grpo/Train_sh_files/Curriculum-based_RL_math_resize/stage1_math_resize.sh
```

### Stage 2: Choice
The placeholders are included for you to replace with your actual paths.
```bash
   torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12348" \
   /Curr_REFT/train_code/grpo/open-r1-multimodal/src/open_r1/Stage2_choice_math_resize.py \
    --deepspeed /mnt/tenant-home_speed/dhl/VLM-R1-main/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir /Stage2_Multi_choice/Stage2_Qwen2.5-VL-3B-GRPO-3_tasks_3000 \ # output
    --model_name_or_path /Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000/checkpoint-500 \ # path/to/Stage1_output
    --dataset_name /data_config/stage2_multi_choice/3_tasks_3000.yaml \  # data_config
    --image_root /path/to/image_root \   #image_data_root
    --max_prompt_length 2048 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
```
After replacing these placeholders with your actual paths, you can run the stage2 script using:
```python
sh /Curr_REFT/train_code/grpo/Train_sh_files/Curriculum-based_RL_math_resize/stage2_math_resize.sh  
```

### Stage 3: Open
The placeholders are included for you to replace with your actual paths.
```bash
 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    /Curr_REFT/train_code/grpo/open-r1-multimodal/src/open_r1/Stage3_open_math_resize.py \
    --deepspeed /mnt/tenant-home_speed/dhl/VLM-R1-main/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir /Stage3_Open/Stage3_Qwen2.5-VL-3B-GRPO-3_tasks_3000 \ # output
    --model_name_or_path /Stage2_Multi_choice/Stage2_Qwen2.5-VL-3B-GRPO-3_tasks_3000/checkpoint-500 \
    --dataset_name /data_config/others/det_classify_math_v2.yaml \    # data_config
    --image_root /path/to/image_root \   #image_data_root
    --max_prompt_length 2048 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
```
After replacing these placeholders with your actual paths, you can run the script for stage3:
```python
sh /Curr_REFT/train_code/grpo/Train_sh_files/Curriculum-based_RL_math_resize/stage3_math_resize.sh
```
3、For Reject-sampling SFT, execute the following scripts (using Qwen2.5-VL-3B as an example):
### Reject-sampling SFT
Activate the SFT environment and run the scripts:
```bash
conda activate swift
```
```bash
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_PIXELS=401408 swift sft \
    --model_type qwen2_5_vl \
    --model /Stage3_Open/Stage3_Qwen2.5-VL-3B-GRPO-3_tasks_3000/checkpoint-2300 \
    --dataset /Curr_REFT/grpo_sft_data/SFT_data/reject/cat_all_formatted.jsonl \  # path/to/reject_json_data
    --train_type full \
    --torch_dtype bfloat16 \
    --max_steps 2000 \
    --max_length 8192 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-7 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --freeze_vit False \
    --save_only_model true \
    --save_total_limit 25 \
    --logging_steps 5 \
    --output_dir /mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/small_learn_rate_Output \
    --warmup_ratio 0.05 \
    --system 'You are a helpful assistant' \
    --deepspeed zero3 \
```


## Evalution

### Jsonl Data for Evaluation
The dataset includes both in-domain and out-of-domain test files for comprehensive evaluation:

**In-Domain Test Files**

| Task Type | File Path |
|-----------|-----------|
| Detection | `/grpo_sft_data/grpo_data/train/open/coco_3k_2task/detection_coco_test.jsonl` |
| Classification | `/grpo_sft_data/grpo_data/train/open/coco_3k_2task/classify_v2_coco_test.jsonl` |
| Math | `/grpo_sft_data/grpo_data/train/open/openr1_8k/math_math_test.jsonl` |

**Out-of-Domain Test Files**

| Task Type | File Path |
|-----------|-----------|
| Detection | `/grpo_sft_data/grpo_data/test/Refgta/refgta_subsample_resize.json` ([RefGTA Image Download](https://drive.google.com/drive/folders/1pcdwA--xSAkbsOwjqhhyXMRZH7_sjQXU)) |
| Classification | `/grpo_data/test/pascal/classify_pascal_voc_test.jsonl` |
| Math | `/grpo_sft_data/grpo_data/test/superclever/superclevr_test200_counting_problems.jsonl`  ([Superclevr Image Download](https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip)) |

These test files are used to evaluate model performance on both familiar (in-domain) and unfamiliar (out-of-domain) data distributions, providing a comprehensive assessment of the model's generalization capabilities.


### Evaluation Scripts
The evaluation uses the same environment configuration as training: [R1-V](https://github.com/ding523/Curr_REFT/blob/main/requirements_for_R1_V.txt)

#### Pass@k Evaluation Script
![combined_pass_at_k_deepseek](https://github.com/user-attachments/assets/93af594b-630a-4ca9-98bb-8a1ce6d8b4c3)
```bash
python Curr_ReFT/eval/pass_at_k.py \
    --input_file /mnt/tenant-home_speed/dhl/VLM-R1-main/Train_sh_files/fig1/pass@k/test_samples/choice/choice_samples.jsonl \
    --task_type "choice" \  # 这个会影响存储的output_dir
    --output_dir /mnt/tenant-home_speed/dhl/VLM-R1-main/Train_sh_files/fig1/pass@k/test_samples/choice \
    --k_values "1,2,4,16,32,64,128,256" \   # 需要测试的k值 用,分隔开即可   
    --model_url 'http://10.55.33.23:30252/v1/chat/completions' \  # 改成你待测模型部署的api链接
    --model_name 'Qwen2.5-VL-7B-Instruct' \  
    --judge_url 'http://10.55.33.23:31203/v1/chat/completions' \     # 裁判模型调用链接
    --judge_model 'NTele-72B-V3' \
    --threads 32    # 并发数  最大32
```

#### Main Evaluation Script
- `/Curr-ReFT/src/eval/muti_process_eval.py`: Multi-GPU evaluation script for most in-domain and out-of-domain tests across all three tasks. (we provide a [shell script](https://github.com/ding523/Curr_REFT/blob/main/eval/Test_multi_GPU.sh).)

#### Task-Specific Scripts
For out-of-domain testing of detection and math tasks, we provide specialized scripts:

- `/Curr-ReFT/src/eval/single_process_eval_for_refgta.py`: Multi-GPU script for RefGTA dataset (out-of-domain detection), includes additional coordinate transformations. Note: We recommend creating a shell script to test multiple checkpoints sequentially ([Test_refgta.sh](https://github.com/ding523/Curr_REFT/blob/main/eval/Test_refgta.sh)).

#### Base Model Evaluation
- `/Curr-ReFT/src/eval/muti_process_eval_for_base_model.py`: Multi-GPU script for generating comparative results with base models


```


## Dataset
### Curriculum Reinforcement Learning Data

The Curriculum Reinforcement Learning Data spans three distinct multimodal tasks:

**Visual Detection**  
- Training: 3,000 images sampled from RefCOCO
- In-domain Testing: 1,000 images from RefCOCO
- Out-of-domain Testing: 1,000 samples from RefGTA for evaluating object localization capabilities

**Visual Classification**
- Training: 3,000 images from RefCOCO and RefCOCOg
- In-domain Testing: 1,000 samples from the same sources
- Out-of-domain Testing: 1,000 samples from Pascal-VOC for evaluating visual categorization ability

**Multimodal Mathematical Reasoning**
- Training: 3,000 samples covering geometry proofs and visual math problems from Math360K and Geo170K
- In-domain Testing: 1,000 samples from the same sources
- Out-of-domain Testing: 500 samples from CLEVER-70k-Counting

**Note**: Files with suffixes "_train" and "_val" are used for training, while files with the "_test" suffix are used exclusively for testing. The "_val" files are not used for validation but are incorporated into the training data, which does not affect the integrity of in-domain testing results.

**Data Format**

Reasoning problems are stored in JSON format, with each row containing the following fields:

- `problem`: The visual reasoning question presented to the model
- `solution`: The ground truth answer in the following formats:
   - Stage 1 (True/False) and Stage 2 (Multiple Choice):``answer>ground truth answer</answer>``
   - Stage 3 (Open-ended): ``<thinking>Thinking</thinking><answer>ground truth answer</answer>``
- `task`: The task category identifier ("detection", "classify", or "math")
- `image`: Path to the associated image file (requires modification to match your local directory structure)


**Rejected Sample based Self-improvement Data**

Rejected Sample based Self-improvement Data comprises 1,520 high-quality examples across diverse domains: pure text mathematics, science, multimodal mathematics, and general knowledge. This dataset was meticulously curated using GPT-4-O as a reward model to evaluate generated responses against multiple criteria: accuracy, logical consistency, format compliance, and linguistic fluency. Responses were quantitatively assessed on a 0-100 scale, with only those surpassing a threshold of 85 being integrated into the enhanced dataset alongside their corresponding queries. 

1. **Mathematics Domain** (700 examples):
   - **Multimodal Data** (300 examples):
     * Geometry3K_MathV360K (100 examples)
     * Geo170k_qa (100 examples)
     * Geomverse (100 examples)
   - **Pure Text Data**:
     * SK1.1 Math Problems (400 examples)

2. **Science Domain** (320 examples):
   - **Multimodal Data** (220 examples):
     * Scienceqa_cauldron (100 examples)
     * Scienceqa_nona_context (120 examples)
   - **Pure Text Data**:
     * SK1.1 Science Problems (100 examples)

3. **General Knowledge Domain** (500 multimodal examples):
   * Illava_cot_100k (300 examples)
   * Visual7w (100 examples)
   * VSR (100 examples)

**Data Format**

```bash
{
  "messages": [
    {
      "role": "user",
      "content": "Question"
    },
    {
      "role": "assistant",
      "content": "<answer>The ground truth answer</answer>"
    }
  ]
}
```
