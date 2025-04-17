
import multiprocessing
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import pandas as pd
from math_verify import parse, verify
import argparse



# from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# gte_model = SentenceTransformer('/mnt/tenant-home_speed/dhl/RL_VL3B/model/GTE-en-large')
# print('GTE_modle_init')

# 1. 图像像素  提速 
parser = argparse.ArgumentParser(description="Your script description here.")
    
    # 添加参数
parser.add_argument('--sample_num', type=int, default=1000,   # json内 选取多少个样本测 全测1000 实验10
                    help='Number of samples to use (default: 10)')

parser.add_argument('--BSZ', type=int, default=80,
                    help='Batch size (default: 32)')

parser.add_argument('--test', type=bool, default=False,
                    help='Batch size (default: 32)')

parser.add_argument('--save_steps', type=int, default=50,
                    help='Steps interval for saving model (default: 100)')

parser.add_argument('--task_list', nargs='+', default=['detection'],
                    help='List of tasks to perform, e.g., detection classify')

parser.add_argument('--model_path', type=str, default='/mnt/tenant-home_speed/dhl/VLM-R1-main/Fix_reward_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou',
                    help='Path(s) to save/load the model (default: /mnt/tenant-home_speed/dhl/VLM-R1-main/Output/Qwen2.5-VL-3B_VQA_det_math_dhl/)')

args = parser.parse_args()

#-------------------------参数初始化 开始

IMAGE_ROOT = "/mnt/tenant-home_speed/dhl/RL_VL3B/test_data/refgta/"


# 给 任务类型 和 是否跨域 返回 测试json路径：
def get_ds_path(task_type, in_domain=True):
    indomain_dict={
        'detection':"",
        'classify':"",
        'math':""
    }  # "/mnt/tenant-home_speed/dhl/RL_VL3B/data/flickr_3k_2task/detection_flickr_test.jsonl"
    outdomain_dict={ 
        'detection':"/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/Refgta/refgta_subsample_resize.json",
    }
    if in_domain:
        return indomain_dict[task_type]
    else:
        return outdomain_dict[task_type]


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

import ast


def extract_answer_items(text):
    """
    从文本中提取 <answer> 标签中的内容，并将其分割成列表
    Args:
        text: 包含 <answer> 标签的文本
    Returns:
        list: 提取的项目列表
    """
    if '[' in text:
        return ast.literal_eval(text)
    try:
        items = re.split(r'[,，、\s]+', text)
        items = [item.strip() for item in items if item.strip()]
        return items
    except Exception as e:
        print(f"Error extracting items: {str(e)}")
        return []
    # else:
    #     try:
    #         # 匹配 <answer> 标签中的内容
    #         pattern = r'<answer>(.*?)</answer>'
    #         match = re.search(pattern, text,re.DOTALL)
    #         if not match:
    #             print("No answer tag found in gt")
    #             return []
    #         answer_content = match.group(1).strip()
    #         items = re.split(r'[,，、\s]+', answer_content)
    #         items = [item.strip() for item in items if item.strip()]
    #         return items
    #     except Exception as e:
    #         print(f"Error extracting items: {str(e)}")
    #         return []


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


def determine_tasks(model_path):
    """
    根据模型路径确定要执行的任务
    
    Args:
        model_path: 模型路径字符串
    
    Returns:
        list: 要执行的任务列表
    """
    tasks = []
    # 转换为小写以进行不区分大小写的匹配
    path_lower = model_path.lower()
    # 检查各种任务标识符
    if 'det' in path_lower:
        tasks.append('detection')
    if any(x in path_lower for x in ['class', 'classify']):
        tasks.append('classify')
    if 'math' in path_lower:
        tasks.append('math') 
    return tasks

def extract_bbox_answer(content):
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_match = re.search(bbox_pattern, content)
    if bbox_match:
        bbox = [int(float(bbox_match.group(1))), int(float(bbox_match.group(2))), int(float(bbox_match.group(3))), int(float(bbox_match.group(4)))]
        x1, y1, x2, y2 = bbox
        # if all(bbox[i] <= 1 for i in range(4)):
        #     bbox = [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)]
        #     return bbox, True
        return bbox, False
    return [0, 0, 0, 0], False

def iou(box1, box2):
    inter_x1 = max(int(box1[0]), int(box2[0]))
    inter_y1 = max(int(box1[1]), int(box2[1]))
    inter_x2 = min(int(box1[2]-1), int(box2[2]-1))
    inter_y2 = min(int(box1[3]-1), int(box2[3]-1))
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

import glob

def get_checkpoint(directory):
    checkpoint_dirs = glob.glob(os.path.join(directory, "checkpoint-*"))
        
    # 只保留文件夹（排除文件）
    checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]

    # 按照checkpoint号码排序
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
    return [os.path.join(directory,path) for path in checkpoint_dirs]




# for in_domain in [True,False]:

def run_test_model(model,processor,model_path,gpu_id,task_list, sample_num, BSZ):
    json_id = int(model_path.split('/')[-1].split('-')[-1])
    MODEL_PATH = f"{model_path}"
    print(f'start_mission {model_path}')
    acc_dict = {}
    acc_dict[int(json_id)] = [
            {
            'detection_in':[],'classify_in':[],'coco_vqa_in':[],'math_in':[],
            'detection_out':[],'classify_out':[],'coco_vqa_out':[],'math_out':[]
            }
        ]
    # 只测试域外
    for in_domain in [False]: 
        # 主函数
        for task_type in task_list: # math 额外测
            single_task_accuracy=[]
            # item_add是类型
            item_add = task_type+'_in' if in_domain else task_type+'_out'
            
            
            ds_path= get_ds_path(task_type,in_domain)
            if ds_path == '':
                print(f'skip {item_add}')
                continue
            print('ds_path',ds_path)
            refgta_flag = 1 if item_add=='detection_out' else 0
            print('refgta_flag',refgta_flag)
            
            
            output_dir = f"/mnt/tenant-home_speed/dhl/VLM-R1-main/Detection_out_Test/{model_path.split('/')[-4]}/Stage3_Open/{model_path.split('/')[-2].replace('Qwen2.5-VL-3B-','')}"
            print('json_output_dir',output_dir)
            os.makedirs(output_dir, exist_ok=True)
            json_output_path=os.path.join(output_dir,item_add,str(model_path.split('/')[-1].split('-')[-1])+'.json')
            
            if os.path.exists(json_output_path): #说明已经测过 跳过这个checkpoint
                continue
                print(f'json_output_path:{json_output_path}已存在 跳过.')
            print('in_domain:',in_domain,'checkpoint:', json_id)
            
            
            # 加载模型
            #"/mnt/tenant-home_speed/dhl/VLM-R1-main/Test/logs/Class_coco2014_epoch_100.json"

            random.seed(42)
            #-------------------------参数初始化 结束
            print(f"Auto Processing...task_type={task_type}, in_domain={in_domain}")
            
            
            # 读json
            if ds_path.endswith(".jsonl"):
                data = []
                with open(ds_path, "r") as json_file:
                    # print('json_path',json_path)
                    for line in json_file:
                        data.append(json.loads(line.strip()))
            elif ds_path.endswith(".json"):
                with open(ds_path, "r") as json_file:
                    data = json.load(json_file)
            
            random.shuffle(data)
            
            # 定义不同任务的问题模板
            QUESTION_TEMPLATE_DETECTION = (
                "{Question} First output your thinking process in <think> </think> tags and then provide the the final answer as bounding box coordinates in <answer> </answer> tags. Output the final answer in JSON format.")
            QUESTION_TEMPLATE_VQA = (
                "{Question} First carefully analyze the image and describe your observations in <think> </think> tags and then poutput the final answer in <answer> </answer> tags. Your answer should be a flowing description using complete sentences, not a list or structured format. The final answer should directly address the question.")
            QUESTION_TEMPLATE_NORMAL = (
                "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.")
            def make_conversation_image(x,task):
                if task=='detection':
                    prompt=QUESTION_TEMPLATE_DETECTION.format(Question=x["problem"])
                elif task=='coco_vqa':
                    prompt=QUESTION_TEMPLATE_VQA.format(Question=x["problem"])
                else:
                    prompt=QUESTION_TEMPLATE_NORMAL.format(Question=x["problem"])
                return prompt
            
            print('len_data',len(data))
            data = data[:sample_num]
            messages = []

            for x in data:
                # image_path = os.path.join(IMAGE_ROOT, x['image'])
                #image_path=x['image']
                try:
                    image_path = x['image']
                    if refgta_flag==1:
                       image_path = os.path.join(IMAGE_ROOT, x['image']).replace('final','final_resize',1)
                    #    print('image_path',image_path)
                except TypeError:
                    print(type(x))  # 应该输出 <class 'dict'> 或者其他预期的数据类型
                    print(x)        # 查看 x 的实际内容
                    print("Error: Expected a dictionary with key 'image', but got a string instead.")
                    # 根据实际情况处理错误
                
                
                message = [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": f"file://{image_path}"
                        },
                        {
                            "type": "text",
                            "text": make_conversation_image(x,task_type)
                        }
                    ]
                }]
                messages.append(message)
                

            all_outputs = []  # List to store all answers

            # Process data
            for i in tqdm(range(0, len(messages), BSZ)):
                batch_messages = messages[i:i + BSZ]
            
                # Preparation for inference
                text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
                
                image_inputs, video_inputs = process_vision_info(batch_messages)
                print()
                inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(f"cuda:{gpu_id}")

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                batch_output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                all_outputs.extend(batch_output_text)
                # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

            final_output = []
            correct_number = 0
            for input_example, model_output in zip(data, all_outputs):
                correct = 0

                original_output = model_output
                bbox = process_answer(model_output,'detection')
                    # print('处理后的结果:------------------------')
                if bbox==[0,0,0,0]:
                    bbox,_=extract_bbox_answer(original_output)
                
                ground_truth=input_example.get('solution')  # refgta的solution不包含<answer> 所以不能process solution
                ground_truth_normalized = input_example.get("normalized_solution")
                ground_truth_resize=input_example.get('resize_solution')
                
                print('bbox',bbox)
                print('ground_truth_resize',ground_truth_resize)
                
                if iou(bbox, ground_truth) > 0.5:
                    correct = 1
                elif iou(bbox, ground_truth_normalized) > 0.5:
                    correct = 1
                elif iou(bbox, ground_truth_resize) > 0.49:
                    correct = 1
                print('iou',iou(bbox, ground_truth_resize))
                print('correct',correct)
                       
                   
                correct_number += correct
                
                # Create a result dictionary for this example
                result = {
                    'question': input_example['problem'],
                    'ground_truth': ground_truth,
                    'model_output': original_output,
                    'extracted_answer': model_answer,
                    'correct': correct
                }
                final_output.append(result)

            # Calculate and print accuracy
            accuracy = correct_number / len(data) * 100
            print(f"\nAccuracy of {json_id} 测试--task_type:{task_type} : {accuracy:.2f}%")
            single_task_accuracy.append(accuracy)
            
            # json的output 
            os.makedirs(os.path.join(output_dir, item_add), exist_ok=True)
            
            # if first_json:
            #     save_steps = int(model_path.split('/')[-1].split('-')[-1])
            #     first_json=False
            with open(json_output_path, "w") as f:
                json.dump({
                    'accuracy': accuracy,
                    'results': final_output
                }, f, indent=2)
            print(f"Results saved to {json_output_path}")
            print("-"*100)
            # 更新 accuracy_list

            # 保存总表
            # print(f"Appending accuracy {accuracy} for checkpoint {json_id} under key {item_add}")
            
            acc_dict[int(json_id)][0][item_add].append(accuracy)
            # print('acc_dict',acc_dict)
            # print('acc_dict[int(json_id)][0]',acc_dict[int(json_id)][0])
            
            # acc_dict[int(json_id)] = [
            #         {
            #         'detection_in':[],'classify_in':[],'coco_vqa_in':[],'math_in':[],
            #         'detection_out':[],'classify_out':[],'coco_vqa_out':[],'math_out':[]
            #         }
            #     ]
            #print(acc_dict[json_id][0][item_add])
            #print(acc_dict)
    del model
    return {'gpu_id':gpu_id,'acc_dict':acc_dict}


def sort_dict_by_keys(input_dict):
    """
    按键排序字典。
    :param input_dict: 输入字典
    :return: 排序后的字典
    """
    return dict(sorted(input_dict.items()))



def handle_result(result):
    """定义一个简单的回调函数来处理结果"""
    print(f"Completed with result: {result}")


def load_model(j):
    """加载模型并返回模型、处理器等"""
    model_path = j['model_path']
    gpu_id = j['gpu_id']
    print(f'load model {j}')
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu_id}",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print('finish load model')
    
    if 'sft' in model_path.lower():  # 使用小写比较来避免大小写问题
        tokenizer = processor.tokenizer  # 获取processor中的tokenizer
        tokenizer.padding_side = 'left'
    
    return model, processor, model_path, gpu_id


from multiprocessing import Pool, current_process
import concurrent.futures


from multiprocessing import Process, Manager, Lock

# checkpoint_list = checkpoint_list
 

if __name__ == '__main__':
    multiprocessing.freeze_support()
        
    sample_num = args.sample_num

    

    first_json = True

    BSZ=args.BSZ
    save_steps = args.save_steps
    
    checkpoint_list = get_checkpoint(args.model_path)
    manager = multiprocessing.Manager()
    

    # 任务类型列表 
    # ['detection','classify','math']
    # task_list=determine_tasks(args.model_path)
    if not hasattr(args, 'task_list') or args.task_list is None:
        raise ValueError("The --task_list argument must be provided.")
    
    task_list=args.task_list
        
    multi_task=False
    if len(task_list)>1:
        multi_task=True
    print('task_list',task_list)

    test=args.test

    # 接力测试
    
    
    # /mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward
   
    # checkpoint_list =  \
    #     [  '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-900', \
    #         '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-1000', \
    #      '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-1100',  \
    #      '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-1200',  \
    #          '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-1300',  \
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-1400',  \
    #         '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-1500',  \
    #      '/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou_fix_reward/checkpoint-1600']
        
    #  /mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450
    # checkpoint_list =  \
    #     [  '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-100', \
    #         '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-200', \
    #      '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-300',  \
    #      '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-400',  \
    #          '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-500',  \
    #     '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-600',  \
    #         '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-700',  \
    #      '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-800']
        
 
    # checkpoint_list = \
    #     ['/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-1700',  
    #     '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-1800',  
    #     '/mnt/tenant-home_speed/dhl/RL_VL3B/sft_data/v1-20250224-102450/checkpoint-1900']
                
    print('checkpoint_list', checkpoint_list) # type: ignore
    # 加两个 如果每隔100 steps跑一次
    # 如果已经有这个json文件就跳过 本次测试


    multiprocessing.set_start_method('spawn',force=True)
    dict_list = [{'gpu_id':i%8,'model_path':model_path} for i,model_path in enumerate(checkpoint_list)]
    Max_workers=8

    lock = Lock()
    async_results = []
    print(f'total file check num:{len(dict_list)}')
    # multiprocessing.set_start_method('spawn', force=True)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=Max_workers) as executor:
    all_ready_ok_gpu = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=Max_workers) as executor:
        async_results = []
        
        for j in dict_list:
            # 加载模型
            if all_ready_ok_gpu:
                gpu_id_free = all_ready_ok_gpu.pop()
                j['gpu_id'] = gpu_id_free
            model, processor, model_path, gpu_id = load_model(j)
            
            # 检查当前正在运行的任务数量
            while len(async_results) >= Max_workers:
                # 等待某个任务完成
                done, async_results = concurrent.futures.wait(async_results, return_when=concurrent.futures.FIRST_COMPLETED)
                async_results = list(async_results)
                for future in done:
                    try:
                        result = future.result()
                        all_ready_ok_gpu.append(result['gpu_id'])
                        print(f"Task completed with result: {result}")
                    except Exception as e:
                        print(f"A task generated an exception: {e}")
            
            # 提交任务
            async_result = executor.submit(run_test_model, model, processor, model_path, gpu_id, task_list, sample_num, BSZ)
            async_results.append(async_result)
        
        # 等待所有任务完成
        results = [future.result() for future in concurrent.futures.as_completed(async_results)]
        
        print("All results:", results)
        
    acc_dict = {}
    for res in results:
        acc_dict[list(res['acc_dict'].keys())[0]] = list(res['acc_dict'].values())[0]

    ordinary_dict = dict(acc_dict)
    # print(ordinary_dict)
    
        # 对普通字典按键进行排序
    sorted_dict = sort_dict_by_keys(ordinary_dict)

    acc_dict_real = {
        'detection_in':[],'classify_in':[],'coco_vqa_in':[],'math_in':[],
        'detection_out':[],'classify_out':[],'coco_vqa_out':[],'math_out':[]
    }

    # print(sorted_dict)

    for values in sorted_dict.values():
        for item in acc_dict_real.keys():
            if values[0][item]!= []:
                # print(values[0][item])
                acc_dict_real[item].append(values[0][item][0])



    # 画图
    print('accuracy_list',acc_dict_real)
    # if not test:
    #     output_dir = f"/mnt/tenant-home_speed/dhl/VLM-R1-main/Test/{model_path.split('/')[-3]}/{model_path.split('/')[-2].replace('Qwen2.5-VL-3B-','')}/{task_type}"
    # else:
    #     output_dir = f"/mnt/tenant-home_speed/dhl/VLM-R1-main/code_test/{model_path.split('/')[-3]}/{model_path.split('/')[-2].replace('Qwen2.5-VL-3B-','')}/{task_type}"
    
    output_dir = f"/mnt/tenant-home_speed/dhl/VLM-R1-main/Test/{model_path.split('/')[-3]}/{model_path.split('/')[-2].replace('Qwen2.5-VL-3B-','')}"
    os.makedirs(output_dir, exist_ok=True)
    excel_output_dir=output_dir
    print('output_dir',output_dir)
    # save成excel  保存代码有错

    # iltered_acc_dict = {k: v for k, v in acc_dict.items() if v}
    # item = iltered_acc_dict.keys()[0]
    # iltered_acc_dict['steps'] = []
    # for i in range(1,len(iltered_acc_dict[item])+1):
    #     iltered_acc_dict['steps'].append(i*save_steps)
    # excel_output_path= os.path.join(excel_output_dir,"all_accuracy_results.xlsx")
    # df = pd.DataFrame(iltered_acc_dict)
    # df.to_excel(excel_output_path, index=False)
    # print(f"Saved to {excel_output_path}")

    
    # 删除空的键值对
    filtered_acc_dict = {k: v for k, v in acc_dict_real.items() if v} # 过滤掉列表为空的键值对

    # 准备数据用于DataFrame
    data = {}
    for metric, accuracies in filtered_acc_dict.items():
        temp_list = []
        for step, accuracy in enumerate(accuracies, start=1):
            temp_list.append(accuracy)
        data[metric] = temp_list

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 添加steps列
    steps=[]
    for _check in checkpoint_list:
        steps.append(_check.split('-')[-1])
    df['steps'] = steps

    # 将DataFrame保存为Excel文件

    df.to_excel(os.path.join(excel_output_dir,'all_accuracy_results.xlsx'), index=False)

    print(f"数据已成功保存到 {os.path.join(excel_output_dir,'all_accuracy_results.xlsx')}")


    #  plot(accuracy_list)





