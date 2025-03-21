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
        QUESTION_TEMPLATE_DETECTION = (
            "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format in <answer> </answer> tags.")
        QUESTION_TEMPLATE_VQA = (
            "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags")
        QUESTION_TEMPLATE_NORMAL = (
            "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.")
        # This is only for Grounding task
        # QUESTION_TEMPLATE_DETECTION = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        # QUESTION_TEMPLATE_NORMAL = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
        def make_conversation_image(example,task):
            
            if task=='detection':
                prompt=QUESTION_TEMPLATE_DETECTION.format(Question=example["problem"])
            elif task=='coco_vqa':
                prompt=QUESTION_TEMPLATE_VQA.format(Question=example["problem"])
            else:
                prompt=QUESTION_TEMPLATE_NORMAL.format(Question=example["problem"])
            
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
        else:
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



def iou_reward(completions, solution, **kwargs):
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
    
    
    def process_answer(content,task_type):
        # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
        # answer_tag_pattern = r'<answer>(.*?)</answer>'
        # content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        # if content_answer_match:
        
         # 使用赋值形式处理不同情况
        if full_match := re.search(r'<answer>(.*?)</answer>', content, re.DOTALL):
            answer = full_match.group(1).strip()
        elif partial_match := re.search(r'<answer>(.*?)($|<|[^<]$)', content, re.DOTALL):
            print("Warning: Found incomplete answer tags")
            answer = partial_match.group(1).strip()
        else:
            print('no <answer\> tag')
            answer = content.strip()
        if task_type=='classify' or task_type=='classify':  # 其实只有 detection 才用 process answer
            return answer   
        # 从 <answer> <answer>中获取答案字段   detection任务需要额外获取box  
    
        if task_type=='detection':
            # content_answer = content_answer_match.group(1).strip()
            # content_answer=content_answer.replace(" ","")
            # print(f'content_answer is {content_answer}')
            answer=answer.replace(" ","")
            bbox_pattern = r'(?:\{)?.*\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\].*(?:\})?'
            bbox_match = re.search(bbox_pattern, answer,re.DOTALL)
            # print(bbox_match)
            if bbox_match:
                bbox = [int(float(bbox_match.group(1))), int(float(bbox_match.group(2))), int(float(bbox_match.group(3))), int(float(bbox_match.group(4)))]
                x1, y1, x2, y2 = map(int, bbox)
                # print('x1, y1, x2, y2',x1, y1, x2, y2)
                return bbox
            else:
                print(f'task_type: {task_type} Empty matching! {content}')
                return [0, 0, 0, 0]
        else:
            return answer


    def process_solution(content,task_type):
        # 首先匹配<answer>标签中的内容
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, content,re.DOTALL)
        
        if task_type=='detection':
            if answer_match:
                bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
                bbox_match = re.search(bbox_pattern, answer_match.group(1),re.DOTALL)
                if bbox_match:
                    # 转换为整数
                    bbox = [int(float(bbox_match.group(i))) for i in range(1, 5)]
                    return bbox
                else:
                    print('detection ground_truth extracts nothing.')
                    return [0,0,0,0]
        else:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', content,re.DOTALL)
            gt = sol_match.group(1).strip() if sol_match else content.strip()
            print(f'task_type:{task_type} gt:{gt}')
            return gt
        
    def calculate_list_iou_reward(pred_list, target_list):
        """
        计算两个列表的交并比（类似IoU）
        
        Args:
            pred_list: 预测的列表
            target_list: 目标列表
            
        Returns:
            float: 交集元素个数 / 并集元素个数
        """
        # 转换为集合
        pred_set = set(pred_list)
        target_set = set(target_list)
        
        # 计算交集和并集
        intersection = pred_set.intersection(target_set)
        union = pred_set.union(target_set)
        
        # 计算比值
        iou = len(intersection) / len(union) if union else 0.0
        return iou
    
    def extract_bbox_answer(content):
        if full_match := re.search(r'<answer>(.*?)</answer>', content, re.DOTALL):
            answer = full_match.group(1).strip()
        elif partial_match := re.search(r'<answer>(.*?)($|<|[^<]$)', content, re.DOTALL):
            answer = partial_match.group(1).strip()
        else:
            answer = content.strip()
        bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
        bbox_match = re.search(bbox_pattern, answer)

        if bbox_match:
            bbox = [int(float(bbox_match.group(1))), int(float(bbox_match.group(2))), int(float(bbox_match.group(3))), int(float(bbox_match.group(4)))]
            x1, y1, x2, y2 = bbox
            # if all(bbox[i] <= 1 for i in range(4)):
            #     bbox = [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)]
            #     return bbox, True
            return bbox, False
        return [0, 0, 0, 0], False

    def extract_answer_items(text,is_sol):
        """
        从文本中提取 <answer> 标签中的内容，并将其分割成列表
        Args:
            text: 包含 <answer> 标签的文本
        Returns:
            list: 提取的项目列表
        """
        if not is_sol:
            if '[' in text:
                return ast.literal_eval(text)
            else:
                try:
                    items = re.split(r'[,，、\s]+', text)
                    items = [item.strip() for item in items if item.strip()]
                    return items
                except Exception as e:
                    print(f"Error extracting items: {str(e)}")
                    return []
        else:# 答案
            try:
                # 匹配 <answer> 标签中的内容
                pattern = r'<answer>(.*?)</answer>'
                match = re.search(pattern, text,re.DOTALL)
                if not match:
                    print("No answer tag found in gt")
                    return []
                answer_content = match.group(1).strip()
                items = re.split(r'[,，、\s]+', answer_content)
                items = [item.strip() for item in items if item.strip()]
                return items
            except Exception as e:
                print(f"Error extracting items: {str(e)}")
                return []
            
        
        
    contents = [completion[0]["content"] for completion in completions]
    
    task_list = kwargs.get('task')
    kwargs_dict = dict(kwargs)
    problem_list = kwargs.get('problem')
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    for content, sol, task,_problem in zip(contents, solution,task_list,problem_list):
        reward = 0.0
        # Try symbolic verification first
        print('task',task)
        if task=='detection':
            # print(f'start_detection content is {content} \n sol is {sol}')
            try:
                bbox=process_answer(content,task)
                sol=process_solution(sol,task)
                if bbox==[0,0,0,0]:
                    bbox,_=extract_bbox_answer(content)
                print(f'bbox {bbox}')
                print(f'sol {sol}')
                if iou(bbox, sol) > 0.5:
                    reward = 1.0
                print('reward',reward)
            except Exception:
                pass  # Continue to next verification method if this fails
        elif task=='classify': # classify  soft
            try:
                # print('处理前 answer:',content,'\n')
                # 使用赋值形式处理不同情况
                if full_match := re.search(r'<answer>(.*?)</answer>', content, re.DOTALL):
                    answer = full_match.group(1).strip()
                elif partial_match := re.search(r'<answer>(.*?)($|<|[^<]$)', content, re.DOTALL):
                    print("Warning: Found incomplete answer tags")
                    answer = partial_match.group(1).strip()
                else:
                    answer = content.strip()
                print('处理后 answer:',answer )
                # Extract answer from solution if it has think/answer tags
                ground_truth_list = extract_answer_items(sol,is_sol=True)
                student_answer_list = extract_answer_items(answer,is_sol=False)
                print('ground_truth:',len(ground_truth_list),ground_truth_list,'\n')
                print('student_answer:',len(student_answer_list),student_answer_list,'\n')
                reward = calculate_list_iou_reward(ground_truth_list,student_answer_list)
                print('reward',reward)
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
            '''原版'''
            # try:
            #     # Extract answer from solution if it has think/answer tags
            #     sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            #     ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
            #     # Extract answer from content if it has think/answer tags
            #     content_match = re.search(r'<answer>(.*?)</answer>', content,re.DOTALL)
            #     student_answer = content_match.group(1).strip() if content_match else content.strip()
            #     # Compare the extracted answers
            #     if ground_truth in student_answer:
            #         reward = 1.0
            #     else:
            #         sentences = [student_answer, ground_truth]
            #         embeddings = gte_model.encode(sentences)
            #         sim_cosine = float(cos_sim(embeddings[0], embeddings[1]).cpu().numpy()[0])
            #         # Compare the extracted answers
            #         if sim_cosine > 0.90:
            #             reward = 1.0   
            #         elif sim_cosine > 0.85:
            #             reward = 0.2
            # except Exception:
            #     pass  # Keep reward as 0.0 if both methods fail
        elif task=='math':   
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails
            print('math answer:',parse(content))
            print('math solution:',parse(sol))
            print('reward',reward)
            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:  # 如果 答案出现在回答中也算对 
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                    
                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()
                    
                    # Compare the extracted answers
                    if ground_truth in student_answer or student_answer in ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
        elif task=='coco_vqa':
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol,re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content,re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                sentences = [student_answer, ground_truth]
                embeddings = gte_model.encode(sentences)
                sim_cosine = float(cos_sim(embeddings[0], embeddings[1]).cpu().numpy()[0])

                # Compare the extracted answers
                if sim_cosine > 0.9:
                    reward = 0.2
                if sim_cosine > 0.95:
                    reward = 1.0   
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
