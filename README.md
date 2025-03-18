# Curr_REFT---Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning
- We will update the data, code, and model weight, once the cor finish checking.

This is our official implementation of paper 
> Huilin Deng, Ding Zou, Rui Ma, Hongchen Luo, Yang Cao, Yu Kang (2025)
Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning [Paper in arXiv](https://arxiv.org/abs/2503.07065).

## Introduction
We deeply investigate the R1-like RL in VLM(MLLM), mainly for answering the following questions:
1. Can Rule-based Reinforcement Learning (RL) applied to multimodal mathematical data enhance general capabilities? 
2. Is it feasible to implement Rule-based RL in other multimodal tasks or computer vision (CV) tasks? If so, what specific improvements can be achieved?
3. For small-scale multimodal large models with limited baseline capabilities, can the aforementioned experiences be applicable? If yes, what optimization strategies should be adopted?
4. After the RL phase, is there a possibility of performance degradation in certain aspects of the model? How can the overall training process be refined to address this issue?

## Requirement
1、We implete it based on [MS-Swift](https://github.com/modelscope/ms-swift) 

2、We perform GRPO based on [OpenR1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal/)

3、As a result, the env could refer the above repo, thanks for their great work!

## Usage
1、We organized the code as follows:
- grpo_sft_data
  - grpo_data
  - SFT_data
- train_code
  - grpo
  - sft
  
2、Hence, you only need to fine the right order and run .sh as the paper claims.

## Dataset
We have included the jsonl in the project, and the fig could be download [here](http...)

## Result Update


## Citation




