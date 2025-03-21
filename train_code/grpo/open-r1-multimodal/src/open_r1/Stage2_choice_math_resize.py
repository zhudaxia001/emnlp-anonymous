# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

gte_model = SentenceTransformer('/mnt/tenant-home_speed/dhl/RL_VL3B/model/GTE-en-large')
print('GTE_modle_init')

def custom_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    # bf16
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    if position_embeddings is None:
        # print("The attention layers in this model are transitioning...")
        # 在这种情况下，rotary_pos_emb就是我们需要的角度值
        freqs = rotary_pos_emb.to(torch.bfloat16)
    else:
        # 如果提供了position_embeddings，我们需要从cos和sin反推角度值
        cos, sin = position_embeddings
        # 使用arctan2从cos和sin计算角度
        freqs = torch.atan2(sin.to(torch.bfloat16), cos.to(torch.bfloat16))

    # 分别对q和k应用旋转位置编码
    q = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), freqs)
    k = apply_rotary_pos_emb_flashatt(k.unsqueeze(0), freqs)
    
    q = q.squeeze(0).to(torch.bfloat16)
    k = k.squeeze(0).to(torch.bfloat16)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output).to(torch.bfloat16)
    return attn_output



Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments,val=False):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                if val:
                    datasets = yaml_data.get("val_datasets")
                else:
                    datasets = yaml_data.get("train_datasets")
                
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999             
                
                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            # print('json_path',json_path)
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        # FIXME
        
        # 定义不同任务的问题模板
        # 定义不同任务的问题模板
        QUESTION_TEMPLATE_others = (
            "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Ensure your final answer is exactly one of the provided option letters without any additional text.")
        
        # 分类任务 允许多选
        QUESTION_TEMPLATE_Classify = (
            "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Based on the given options, select all options that accurately describe what is shown in the image. Ensure your final answer is exactly one or more of the provided option letters without any additional text.")
        
       
        # This is only for Grounding task
        # QUESTION_TEMPLATE_DETECTION = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        # QUESTION_TEMPLATE_NORMAL = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
        def make_conversation_image(example,task):
                
            prompt=QUESTION_TEMPLATE_others.format(Question=example["problem"])
            if task=='classify':
                prompt=QUESTION_TEMPLATE_Classify.format(Question=example["problem"])
            
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            }

        def round_by_factor(x: int, factor: int) -> int:
            """将数字四舍五入到最近的能被factor整除的数"""
            return factor * round(x / factor)

        def floor_by_factor(x: float, factor: int) -> int:
            """将数字向下取整到最近的能被factor整除的数"""
            return factor * math.floor(x / factor)

        def ceil_by_factor(x: float, factor: int) -> int:
            """将数字向上取整到最近的能被factor整除的数"""
            return factor * math.ceil(x / factor)
        
        def smart_resize(
            height: int, 
            width: int, 
            factor: int ,  #IMAGE_FACTOR
            min_pixels: int ,  #MIN_PIXELS
            max_pixels: int , #MAX_PIXELS
            MAX_RATIO: int
            ) -> tuple[int, int]:
            """
            智能调整图片尺寸，保持以下条件：
            1. 高度和宽度都能被factor整除
            2. 总像素数在[min_pixels, max_pixels]范围内
            3. 尽可能保持原始宽高比
            Args:
                height (int): 原始图片高度
                width (int): 原始图片宽度
                factor (int): 尺寸需要被此数整除
                min_pixels (int): 最小像素数限制
                max_pixels (int): 最大像素数限制
            Returns:
                tuple[int, int]: 调整后的(高度, 宽度)
            Raises:
                ValueError: 当图片宽高比超过MAX_RATIO时抛出异常
            """
            # 检查宽高比是否在允许范围内
            if max(height, width) / min(height, width) > MAX_RATIO:
                raise ValueError(
                    f"宽高比不能超过{MAX_RATIO}, 当前比例为{max(height, width) / min(height, width)}"
                )
            
            # 初步调整：确保高度和宽度都能被factor整除
            h_bar = max(factor, round_by_factor(height, factor))
            w_bar = max(factor, round_by_factor(width, factor))
            
            # 如果总像素数超过最大限制
            if h_bar * w_bar > max_pixels:
                # 计算需要的缩放比例
                # beta = sqrt(原始面积/最大允许面积)
                beta = math.sqrt((height * width) / max_pixels)
                
                # 按比例缩小，并确保能被factor整除
                # 使用floor_by_factor向下取整，避免超过max_pixels
                h_bar = floor_by_factor(height / beta, factor)
                w_bar = floor_by_factor(width / beta, factor)
                
            # 如果总像素数小于最小限制
            elif h_bar * w_bar < min_pixels:
                # 计算需要的放大比例
                # beta = sqrt(最小所需面积/原始面积)
                beta = math.sqrt(min_pixels / (height * width))
                
                # 按比例放大，并确保能被factor整除
                # 使用ceil_by_factor向上取整，确保达到min_pixels
                h_bar = ceil_by_factor(height * beta, factor)
                w_bar = ceil_by_factor(width * beta, factor)
        
            return h_bar, w_bar
        
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")

             # 检查是否是数学任务且图片尺寸超过限制
            width, height = image.size
            current_pixels = width * height
            Max_pixels= 410000
            print('task',example['task'],f'w,d,n:{width},{height},{current_pixels}')
            
            if example['task'] == 'math' and current_pixels > Max_pixels:
                print(f"Resizing math task image from {width}x{height} ({current_pixels} pixels)")

                try:
                    new_height, new_width = smart_resize(
                        height=height,
                        width=width,
                        factor=1, # 图片需要被整除
                        min_pixels= 1000,
                        max_pixels= Max_pixels,
                        MAX_RATIO=12
                    )
                     # image = image.resize((new_width, new_height))
                    # 使用resize来获得精确的尺寸
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"Resized to {new_width}x{new_height} ({new_width * new_height} pixels)")
                    actual_width, actual_height = image.size
                    print(f'Actual size after resize: {actual_width}×{actual_height} ({actual_width * actual_height} pixels)')
                except ValueError as e:
                    print(f"Warning: Image {image_path} resize failed: {e}")
                    # 可以选择跳过这张图片或使用原始尺寸
            elif current_pixels > 410000:
                print('task',example['task'],'current_pixels > 410000')
                try:
                    new_height, new_width = smart_resize(
                        height=height,
                        width=width,
                        factor=1, # 图片需要被整除
                        min_pixels= 1000,
                        max_pixels= 420000,
                        MAX_RATIO=12
                    )
                     # image = image.resize((new_width, new_height))
                    # 使用resize来获得精确的尺寸
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"Resized to {new_width}x{new_height} ({new_width * new_height} pixels)")
                    actual_width, actual_height = image.size
                    print(f'Actual size after resize: {actual_width}×{actual_height} ({actual_width * actual_height} pixels)')
                except ValueError as e:
                    print(f"Warning: Image {image_path} resize failed: {e}")

        else:
            print(f'None image file!!!image_path:{image_path}')
            image = None
        
        # print('problem',example['problem'])
        # print('image',example['image'])
        # print('solution',example['solution'])
        # print('task',example['task'])
        # print('prompt',make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'])
   
        return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example,task=example['task'])['prompt'] if 'image' in example else make_conversation(example)['prompt'],
            'task':  example['task']
        }

#  重点改进！
'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''



def iou_reward(completions, solution, **kwargs): #计算 准确度
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
        
    def extract_answer_content(text):
        """
        提取答案内容，处理各种可能的格式
        """
        # 尝试标准格式 <answer>...</answer>
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 尝试不标准格式 <answer>...<answer>
        match = re.search(r'<answer>(.*?)(?:<answer>|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 2. 尝试只有开始标签的格式 <answer>...
        match = re.search(r'<answer>(.*?)$', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 3. 尝试重复标签格式 <answer>...<answer>
        match = re.search(r'<answer>(.*?)(?:<answer>|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()

    def extract_options(text):
        """
        从文本中提取选项（A-E, a-e）
        只提取独立的字母选项，避免匹配单词中的字母
        """
        # 转换为大写并移除空格
        try:
            text = text.upper().strip()
            
            # 使用更精确的正则表达式匹配选项
            # 匹配模式：字母前后是空格、逗号、句号等分隔符
            
            options = re.findall(r'(?<![A-Za-z])([A-E])(?![A-Za-z])', text)
        
            # 去重并保持顺序
            return list(dict.fromkeys(options))
        except:
            print('None!')
            return None

    def get_final_answer(text):
        """
        完整的答案提取过程
        1. 提取<answer>标签内容
        2. 从内容中提取选项
        3. 返回格式化的答案
        """
        try:
            # 先提取标签内容
            content = extract_answer_content(text)
            # 再提取选项
            options = extract_options(content)
            # 返回格式化的答案
            return ','.join(options)
        except:
            return None

        
    contents = [completion[0]["content"] for completion in completions]
    
    task_list = kwargs.get('task')
    kwargs_dict = dict(kwargs)
    problem_list = kwargs.get('problem')
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, task,_problem in zip(contents, solution,task_list,problem_list):
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
            # print('parse result',float(verify(answer, parse(solution))))
            if get_final_answer(content) !=None:
                result = (get_final_answer(content)==get_final_answer(sol))
                reward = 1 if result else 0

            print('student',get_final_answer(content),'solution',get_final_answer(sol),'reward',reward)
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Task: {task}\n")
                f.write(f"Problem: {_problem}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    tasklist = kwargs.get('task')
    matches = []
    for task,completion in zip(tasklist,completions):
        content = completion[0]["content"]
        if task=='detection':
            pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        else:
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        # completion_contents = [completion[0]["content"] for completion in completions]
        matches.append(re.fullmatch(pattern, content, re.DOTALL))
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": iou_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # print("reward_funcs:", reward_funcs)

    # Load the dataset
    train_dataset  = LazySupervisedDataset(script_args.dataset_name, script_args,val=False)
    val_dataset = LazySupervisedDataset(script_args.dataset_name, script_args,val=True)
    
    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    # trainer = trainer_cls(
    #     model=model_args.model_name_or_path,
    #     reward_funcs=reward_funcs,
    #     args=training_args,
    #     train_dataset=dataset,
    #     eval_dataset=None,
    #     peft_config=get_peft_config(model_args),
    #     attn_implementation=model_args.attn_implementation,
    #     max_pixels=script_args.max_pixels,
    #     min_pixels=script_args.min_pixels,
    #     torch_dtype=model_args.torch_dtype,
    # )
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
