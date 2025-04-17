---
license: apache-2.0
task_categories:
- question-answering
- multiple-choice
- true-false
size_categories:
- 1K<n<10K
---
# Evaluetion&Curr-ReFT-data

[\[📂 GitHub\]](https://github.com/ding523/Curr_REFT)[\[📝 Paper\]](https://arxiv.org/pdf/2503.07065)
[\[🤗 HF Dataset\]](https://huggingface.co/datasets/ZTE-AIM/Curr-ReFT-data)  [\[🤗 HF-Model: Curr-ReFT-3B\]](https://huggingface.co/ZTE-AIM/3B-Curr-ReFT) 
[\[🤗 HF-Model: Curr-ReFT-7B\]](https://huggingface.co/ZTE-AIM/7B-Curr-ReFT) 


## Dataset Overview

Curr-ReFT-data contains training data for both stages of the Curr-ReFT methodology. The proposed Curr-ReFT post-training paradigm consists of two consecutive training stages: 1. Curriculum Reinforcement Learning: Gradually increasing task difficulty through reward mechanisms that match task complexity. 2. Rejected Sample based Self-improvement: Maintaining the foundational capabilities of the LLM model.

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
* SFT_data: The "reject" folder contains JSON files for Rejected Sample based Self-improvement, while the "sft" folder contains JSONL files for standard SFT (used for comparative experiments in the paper). All images are stored in the "images" folder.

## Test Configuration

The dataset includes both in-domain and out-of-domain test files for comprehensive evaluation:

### In-Domain Test Files

| Task Type | File Path |
|-----------|-----------|
| Detection | `/Curr-ReFT-data/grpo_data/train/open/coco_3k_2task/detection_coco_test.jsonl` |
| Classification | `/Curr-ReFT-data/grpo_data/train/open/coco_3k_2task/classify_v2_coco_test.jsonl` |
| Math | `/Curr-ReFT-data/grpo_data/train/open/openr1_8k/math_math_test.jsonl` |

### Out-of-Domain Test Files

| Task Type | File Path |
|-----------|-----------|
| Detection | `/Curr-ReFT-data/grpo_data/test/Refgta/refgta_subsample_resize.json` ([RefGTA Image Download](https://drive.google.com/drive/folders/1pcdwA--xSAkbsOwjqhhyXMRZH7_sjQXU)) |
| Classification | `/Curr-ReFT-data/grpo_data/test/pascal/classify_pascal_voc_test.jsonl` |
| Math | `/Curr-ReFT-data/grpo_data/test/superclever/superclevr_test200_counting_problems.jsonl`  ([Superclevr Image Download](https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip)) |

These test files are used to evaluate model performance on both familiar (in-domain) and unfamiliar (out-of-domain) data distributions, providing a comprehensive assessment of the model's generalization capabilities.


## Evaluation Scripts

The evaluation uses the same environment configuration as training: [R1-V](https://github.com/ding523/Curr_REFT/blob/main/requirements_for_R1_V.txt)

### Main Evaluation Script
- `/Curr-ReFT/src/eval/muti_process_eval.py`: Multi-GPU evaluation script for most in-domain and out-of-domain tests across all three tasks. (we provide a [shell script](https://github.com/ding523/Curr_REFT/blob/main/eval/Test_multi_GPU.sh).)

### Task-Specific Scripts
For out-of-domain testing of detection and math tasks, we provide specialized scripts:

- `/Curr-ReFT/src/eval/single_process_eval_for_refgta.py`: Multi-GPU script for RefGTA dataset (out-of-domain detection), includes additional coordinate transformations. Note: We recommend creating a shell script to test multiple checkpoints sequentially ([Test_refgta.sh](https://github.com/ding523/Curr_REFT/blob/main/eval/Test_refgta.sh)).

### Base Model Evaluation
- `/Curr-ReFT/src/eval/muti_process_eval_for_base_model.py`: Multi-GPU script for generating comparative results with base models

## Curriculum Reinforcement Learning Data

The Curriculum Reinforcement Learning Data spans three distinct multimodal tasks:

### Visual Detection
- **Training**: 3,000 images sampled from RefCOCO
- **In-domain Testing**: 1,000 images from RefCOCO
- **Out-of-domain Testing**: 1,000 samples from RefGTA for evaluating object localization capabilities

### Visual Classification
- **Training**: 3,000 images from RefCOCO and RefCOCOg
- **In-domain Testing**: 1,000 samples from the same sources
- **Out-of-domain Testing**: 1,000 samples from Pascal-VOC for evaluating visual categorization ability

### Multimodal Mathematical Reasoning
- **Training**: 3,000 samples covering geometry proofs and visual math problems from Math360K and Geo170K
- **In-domain Testing**: 1,000 samples from the same sources
- **Out-of-domain Testing**: 500 samples from CLEVER-70k-Counting

**Note**: Files with suffixes "_train" and "_val" are used for training, while files with the "_test" suffix are used exclusively for testing. The "_val" files are not used for validation but are incorporated into the training data, which does not affect the integrity of in-domain testing results.



### Data Format

Reasoning problems are stored in JSON format, with each row containing the following fields:

- `problem`: The visual reasoning question presented to the model
- `solution`: The ground truth answer in the following formats:
   - Stage 1 (True/False) and Stage 2 (Multiple Choice):``answer>ground truth answer</answer>``
   - Stage 3 (Open-ended): ``<thinking>Thinking</thinking><answer>ground truth answer</answer>``
- `task`: The task category identifier ("detection", "classify", or "math")
- `image`: Path to the associated image file (requires modification to match your local directory structure)



## Rejected Sample based Self-improvement Data


Rejected Sample based Self-improvement Data comprises 1,520 high-quality examples across diverse domains: pure text mathematics, science, multimodal mathematics, and general knowledge. This dataset was meticulously curated using GPT-4-O as a reward model to evaluate generated responses against multiple criteria: accuracy, logical consistency, format compliance, and linguistic fluency. Responses were quantitatively assessed on a 0-100 scale, with only those surpassing a threshold of 85 being integrated into the enhanced dataset alongside their corresponding queries. The self-improvement training process utilizes this high-quality data to optimize the model while preserving its reasoning capabilities, striking a balance between enhancing fundamental skills and maintaining advanced reasoning abilities across diverse domains. The dataset composition is as follows:


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

### Data Format

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



# Institution
- ZTE-AIM
- University of Science and Technology of China

## Model Contact
- huilin_deng@mail.ustc.edu.cn
- zoudinghust@gmail.com
- 214711069@csu.edu.cn
