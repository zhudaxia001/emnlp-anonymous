# Curr_REFT---Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning

- We upload our model weights (https://huggingface.co/ZTE-AIM/3B-Curr-ReFT) (https://huggingface.co/ZTE-AIM/7B-Curr-ReFT) ---2025.03.25
- We upload our data (https://huggingface.co/datasets/ZTE-AIM/Curr-ReFT-data). --- 2025.03.25
- We release our code. --- 2025.03.21
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

  
2、Hence, you only need to fine the right order and run .sh as the paper claims.

## Training

1、Download the code and [grpo_sft_data.zip](https://github.com/ding523/Curr_REFT/blob/main/grpo_sft_data.zip) and organize the code as follows:
- grpo_sft_data
  - grpo_data
  - SFT_data
- train_code
  - grpo
  - sft








## Dataset
We have included the jsonl in the project, and the fig could be download [here](http...)

## Result Update (Results in this version, we use Gpt-3.5t as a judge, which is differenet from the results in paper with Qwen2.5VL-72B judging)

```markdown
| #  | Model                     | AI2D | MMVet | MMBench | MathVista | OCRBench |
|----|---------------------------|------|-------|---------|-----------|----------|
| 1  | Qwen2.5-VL-3B-Instruct    | 74.35| 39.04 | 63.32   | 52.0      | 597      |
| 2  | InternVL2_5-4B            | 75.84| 42.48 | 72.77   | 56.0      | 681      |
| 3  | Qwen2-VL-7B-Instruct      | 79.70| 39.40 | 70.19   | 49.4      | 655      |
| 4  | Qwen2.5-VL-7B-Instruct    | 80.01| 50.69 | 77.92   | 63.8      | 716      |
| 5  | InternVL2_5-8B            | 65.42| 34.17 | 51.98   | 50.1      | 598      |
| 6  | InternVL2_5-26B           | 78.01| 42.11 | 71.56   | 51.4      | 628      |
| 7  | InternVL2_5-38B           | 82.93| 48.58 | 80.07   | 67.6      | 677      |
| 8  | 3B+SFT                    | 75.45| 32.02 | 63.32   | 53.6      | 589      |
| 9  | 3B+RL                     | 76.46| 36.28 | 66.41   | 55.3      | 609      |
| 10 | 3B+Curr-RL                | 77.36| 36.74 | 68.99   | 56.3      | 594      |
| 11 | 3B_Curr-ReFT              | 79.66| 39.95 | 69.27   | 57.9      | 623      |
| 12 | 7B-Curr-ReFT              | 83.16| 49.95 | 80.15   | 65.8      | 727      |




## Citation
@misc{deng2025boostinggeneralizationreasoningvision,
      title={Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning}, 
      author={Huilin Deng and Ding Zou and Rui Ma and Hongchen Luo and Yang Cao and Yu Kang},
      year={2025},
      eprint={2503.07065},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07065}, 
}
