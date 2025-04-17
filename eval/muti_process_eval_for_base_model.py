import multiprocessing
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoModel, AutoTokenizer, AutoProcessor
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
from transformers import AutoTokenizer, AutoModel
import math
from PIL import Image

# from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# gte_model = SentenceTransformer('/mnt/tenant-home_speed/dhl/RL_VL3B/model/GTE-en-large')
# print('GTE_modle_init')

# 1. 图像像素  提速 
parser = argparse.ArgumentParser(description="Your script description here.")
    
    # 添加参数
parser.add_argument('--sample_num', type=int, default=1000,   # json内 选取多少个样本测 全测1000 实验10
                    help='Number of samples to use (default: 10)')

parser.add_argument('--BSZ', type=int, default=60,
                    help='Batch size (default: 32)')

parser.add_argument('--test', type=bool, default=False,
                    help='Batch size (default: 32)')

parser.add_argument('--save_steps', type=int, default=50,
                    help='Steps interval for saving model (default: 100)')

parser.add_argument('--task_list', nargs='+', default=['detection','math','classify'],
                    help='List of tasks to perform, e.g., detection classify')

parser.add_argument('--model_path', type=str, default='/mnt/tenant-home_speed/dhl/VLM-R1-main/Fix_reward_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou',
                    help='Path(s) to save/load the model (default: /mnt/tenant-home_speed/dhl/VLM-R1-main/Output/Qwen2.5-VL-3B_VQA_det_math_dhl/)')

parser.add_argument('--output_dir', type=str, default='/mnt/tenant-home_speed/dhl/VLM-R1-main/Test/Base_model_Output',
                    help='Path(s) to save/load the model (default: /mnt/tenant-home_speed/dhl/VLM-R1-main/Output/Qwen2.5-VL-3B_VQA_det_math_dhl/)')


args = parser.parse_args()
#-------------------------参数初始化 开始



checkpoint_list=['/mnt/tenant-home_speed/AIM/model/Qwen2-VL-2B-Instruct','/mnt/tenant-home_speed/AIM/model/Qwen2-VL-7B-Instruct', 
                        '/mnt/tenant-home_speed/AIM/model/InternVL2-1B','/mnt/tenant-home_speed/AIM/model/InternVL2_5-1B',
                       '/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-3B-Instruct','/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-7B-Instruct',
                       '/mnt/tenant-home_speed/AIM/model/InternVL2_5-4B','/mnt/tenant-home_speed/AIM/model/InternVL2_5-8B']

# 给 任务类型 和 是否跨域 返回 测试json路径：
def get_ds_path(task_type, in_domain=True):
    indomain_dict={
        'detection':"/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/coco_3k_2task/detection_coco_test.jsonl",
        'classify':"/mnt/tenant-home_speed/dhl/RL_VL3B/data/coco_3k_2task/classify_v2_coco_test.jsonl",
        'coco_vqa':"/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/coco2017_vqa/coco_vqa_coco_vqa_test.jsonl",
        'math':"/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/openr1_8k/math_math_test.jsonl"
    }  # "/mnt/tenant-home_speed/dhl/RL_VL3B/data/flickr_3k_2task/detection_flickr_test.jsonl"
    outdomain_dict={ 
        'detection':"", #/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/rec_jsons_processed/refgta_subsample.json
        'classify':"/mnt/tenant-home_speed/dhl/RL_VL3B/data/pascal/classify_pascal_voc_test.jsonl",
        'coco_vqa':"/mnt/tenant-home_speed/dhl/VLM-R1-main/data/llava_pretrain_vqa/llava_vqa_llava_vqa_test.jsonl",
        'math':"/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/geoqa_test_prompts.jsonl"
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
    
    if task_type=='detection':
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, content,re.DOTALL)
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
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', content,re.DOTALL)
            gt = sol_match.group(1).strip() if sol_match else content.strip()
            print(f'task_type:{task_type} gt:{gt}')
            return gt
        except Exception:
            print('no <answer> in solution. counting math')
            return str(content)

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

# 数学选择题
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
    else:
        return text.strip()

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

# print("Steps: ", steps)
def round_by_factor(x: int, factor: int) -> int:
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

# for in_domain in [True,False]:

    
    return messages


def run_test_model(model,processor,model_path,gpu_id,task_list, sample_num, BSZ,output_dir):
    with torch.no_grad():

        MODEL_PATH = f"{model_path}"
        print(f'start_mission {model_path}')
      
        for in_domain in [True,False]: 
        # for in_domain in [False]: 
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
                
            
                print('output_dir',output_dir)
                os.makedirs(os.path.join(output_dir,str(item_add)), exist_ok=True)
                json_output_path=os.path.join(output_dir,str(item_add)+'.json')
               
                if os.path.exists(json_output_path): #说明已经测过 跳过这个checkpoint
                    continue
                    print('json_output_path:',json_output_path,'已存在 跳过.')
                print('in_domain:',in_domain,'checkpoint:', model_path)
                
                
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
                
                data = data[:sample_num]
                messages = []

                for x in data:
                    # image_path = os.path.join(IMAGE_ROOT, x['image'])
                    #image_path=x['image']
                    try:
                        image_path = x['image']
                        if refgta_flag==1:
                            image_path=os.path.join('/mnt/tenant-home_speed/dhl/RL_VL3B/test_data/refgta',image_path)
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
                for i in tqdm(range(0, len(messages), BSZ),desc=f"{gpu_id}--{task_type}"):
                    batch_messages = messages[i:i + BSZ]
                
                    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
                    
                    image_inputs, video_inputs = process_vision_info(batch_messages)
                    
                    temp_images=[]
                    for _image in image_inputs:
                        width, height = _image.size
                        current_pixels = width * height
                        Max_pixels= 410000
                        # print(f'w,d,n:{width},{height},{current_pixels}')
                        
                        if task_type=='math' and current_pixels > Max_pixels:
                            # print(f"Resizing math task image from {width}x{height} ({current_pixels} pixels)")
                            new_height, new_width = smart_resize(
                                height=height,
                                width=width,
                                factor=1, # 图片需要被整除
                                min_pixels= 1000,
                                max_pixels= Max_pixels,
                                MAX_RATIO=40
                            )
                            
                            _image = _image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            # print(f"Resized to {new_width}x{new_height} ({new_width * new_height} pixels)")
                            actual_width, actual_height = _image.size
                            # print(f'Actual size after resize: {actual_width}×{actual_height} ({actual_width * actual_height} pixels)')
                            # except ValueError as e:
                            #     print(f"Warning: Image {image_path} resize failed: {e}")  
                        temp_images.append(_image)
                    
                    image_inputs=temp_images    
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
                    
                    original_output = model_output
                    model_answer = process_answer(original_output,task_type)
                    if refgta_flag==0:
                        ground_truth = process_solution(input_example['solution'],task_type)
                   
                    # Count correct answers 
                    correct = 0
                    if model_answer is not None:
                        if task_type=='detection':        
                            # print(f'start_detection content is {content} \n sol is {sol}')
                            try:
                                bbox=model_answer
                                if bbox==[0,0,0,0]:
                                    bbox,_=extract_bbox_answer(original_output)
                                if refgta_flag==1:
                                    ground_truth=input_example.get('solution',input_example.get('normalized_solution'))  # refgta的solution不包含<answer> 所以不能process solution
                                sol=ground_truth
                                if iou(bbox, sol) > 0.5:
                                    correct = 1
                                if refgta_flag==1:
                                    bbox = [gta_bb*2.2 for gta_bb in bbox]
                                    print(f'bbox answer处理后 {bbox}')
                                    ground_truth_normalized = input_example.get("normalized_solution")
                                    print(f'ground_truth {sol}')
                                    # print('ground_truth_normalized',sol)
                                    if iou(bbox, sol) > 0.5:
                                        correct = 1

                                # print('correct',correct)
                            except Exception:
                                pass  # Continue to next verification method if this fails
                        elif task_type=='classify': 
                            try:
                                if full_match := re.search(r'<answer>(.*?)</answer>', original_output, re.DOTALL):
                                    answer = full_match.group(1).strip()
                                elif partial_match := re.search(r'<answer>(.*?)($|<|[^<]$)', original_output, re.DOTALL):
                                    # print("Warning: Found incomplete answer tags")
                                    answer = partial_match.group(1).strip()
                                else:
                                    answer = original_output.strip()
                                # print('处理后 answer:',answer )
                                # Extract answer from solution if it has think/answer tags
                                ground_truth_list = extract_answer_items(ground_truth)# extract_answer_items函数不提取<answer>标签 ground_truth是已经提取过了 answer是前几行提取过
                                student_answer_list = extract_answer_items(answer)
                                # print('ground_truth:',len(ground_truth_list),ground_truth_list,'\n')
                                # print('student_answer:',len(student_answer_list),student_answer_list,'\n')
                                reward = calculate_list_iou_reward(ground_truth_list,student_answer_list)
                                if reward==1:
                                    correct=1
                                else:
                                    ground_truth_set = set(ground_truth_list)
                                    student_answer_set = set(student_answer_list)
                                    # 检查正确答案集合是否是学生答案集合的子集
                                    if ground_truth_set.issubset(student_answer_set):
                                        correct=1
                                # print('correct',correct)
                            except Exception:
                                print('Classify correct calculation Error!')
                                pass  # Keep reward as 0.0 if both methods fail                        
                        elif task_type=='math':   
                            # Try symbolic verification first
                            # try:
                            print('item_add',item_add)
                            if item_add!='math_out':
                                print('11111111')
                                answer = parse(original_output)
                                print('222222222')
                                if float(verify(answer, parse(input_example['solution']))) > 0:
                                    correct = 1.0
                                    print('parse sucesess!!!')
                                    print('parse result:',f'answer:{answer}','solution:',parse(input_example['solution']))
                            # except Exception:
                            #     print('parse failed...')
                            #     pass  # Continue to next verification method if this fails
                            
                            # print('math answer:',parse(content))
                            # print('math solution:',parse(sol))
                            # print('reward',reward)
                            # If symbolic verification failed, try string matching
                                if correct == 0.0:
                                    # try:  
                                    print('33333333333') 
                                    if get_final_answer(original_output)==get_final_answer(input_example['solution']): # 做个选择题匹配
                                        correct = 1.0
                                        if get_final_answer(original_output)!=None:
                                            print('choice sucesuss!!!')
                                            print('math choice:  ','student_answer',get_final_answer(original_output),'ground_truth',get_final_answer(input_example['solution']))
                                # 如果 答案出现在回答中也算对 
                                # Extract answer from solution if it has think/answer tags
                                ground_truth = extract_answer_content(input_example['solution'])
                                student_answer = extract_answer_content(original_output)
                                print('444444444')
                                if ground_truth in student_answer or student_answer in ground_truth:
                                    correct = 1.0
                                    print('string match sucesuss!!!')
                                if input_example['solution'] in original_output:
                                    correct=1.0
                                    print('string match sucesuss!!!')
                             
                            elif item_add=='math_out':
                                if str(input_example['solution']) in original_output:
                                    correct=1.0
                                    print('string match sucesuss!!!')
                            # print('correct',correct)
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
                print(f"\nAccuracy of {model_path} 测试--task_type:{task_type} : {accuracy:.2f}%")
                single_task_accuracy.append(accuracy)
                
                # json的output 
                os.makedirs(os.path.join(output_dir, item_add), exist_ok=True)
                
                with open(json_output_path, "w") as f:
                    json.dump({
                        'accuracy': accuracy,
                        'results': final_output
                    }, f, indent=2)
                print(f"Results saved to {json_output_path}")
                print("-"*100)
                # 更新 accuracy_list

        
            
    del model
   


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








def load_model(model_path,gpu_id):
    """加载模型并返回模型、处理器等"""
   
    print(f'load model.....',model_path)
    

    print('model_path',model_path)
    print('gpu_id',gpu_id)

    if 'Qwen2.5-VL' in model_path:
        print('loading Qwen2.5-VL.......')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map=f"cuda:{gpu_id}",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        print('finish load Qwen2.5-VL model')
        
        # if 'sft' in model_path.lower():  # 使用小写比较来避免大小写问题
        tokenizer = processor.tokenizer  # 获取processor中的tokenizer
        tokenizer.padding_side = 'left'
    elif 'Qwen2-VL' in model_path:
        print('loading Qwen2-VL.......')
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{gpu_id}",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        print('finish load Qwen2-VL model')
        
    # elif 'InternVL2_5' in model_path or 'InternVL2' in model_path: 
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,trust_remote_code=True,
            device_map=f"cuda:{gpu_id}",).eval().cuda()
        processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True )
        
    
    return model, processor, model_path, gpu_id


from multiprocessing import Pool, current_process
import concurrent.futures


from multiprocessing import Process, Manager, Lock


 


    # checkpoint_list=['/mnt/tenant-home_speed/AIM/model/Qwen2-VL-2B-Instruct','/mnt/tenant-home_speed/AIM/model/Qwen2-VL-7B-Instruct', 
    #         '/mnt/tenant-home_speed/AIM/model/InternVL2-1B','/mnt/tenant-home_speed/AIM/model/InternVL2_5-1B',
    #         '/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-3B-Instruct','/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-7B-Instruct',
    #         '/mnt/tenant-home_speed/AIM/model/InternVL2_5-4B','/mnt/tenant-home_speed/AIM/model/InternVL2_5-8B']
    
    
    # checkpoint_list=['/mnt/tenant-home_speed/AIM/model/Qwen2-VL-2B-Instruct']
    
    # print('checkpoint_list', checkpoint_list) # type: ignore


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    sample_num = args.sample_num
    BSZ = args.BSZ
    save_steps = args.save_steps
    
    if not hasattr(args, 'task_list') or args.task_list is None:
        raise ValueError("The --task_list argument must be provided.")
    
    task_list = [ 'classify', 'math']
    multi_task = True
    test = args.test
    

     # checkpoint_list=['/mnt/tenant-home_speed/AIM/model/InternVL2_5-4B','/mnt/tenant-home_speed/AIM/model/InternVL2_5-8B',
     #        '/mnt/tenant-home_speed/AIM/model/Qwen2-VL-2B-Instruct','/mnt/tenant-home_speed/AIM/model/Qwen2-VL-7B-Instruct', 
    #         '/mnt/tenant-home_speed/AIM/model/InternVL2-1B','/mnt/tenant-home_speed/AIM/model/InternVL2_5-1B',
    #         '/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-3B-Instruct','/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-7B-Instruct'
    #         ]
    # 模型列表
    checkpoint_list=[ '/mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-7B-Instruct',
        '/mnt/tenant-home_speed/AIM/model/InternVL2_5-4B','/mnt/tenant-home_speed/AIM/model/InternVL2_5-8B'       
        ]
   
    print('checkpoint_list', checkpoint_list)
    
    # 结果存储字典
    
    
    # 顺序测试每个模型
    for model_path in checkpoint_list:
        print(f'Testing model: {model_path}')
        output_dir= os.path.join('/mnt/tenant-home_speed/dhl/VLM-R1-main/Test/Base_model_Output',model_path.split('/')[-1])

        os.makedirs(output_dir, exist_ok=True)
        model, processor, model_path, gpu_id=load_model(model_path=model_path,gpu_id=0)
        
        # 运行测试
        result = run_test_model(
            model=model,
            processor=processor,
            model_path=model_path,
            gpu_id=0,  # 固定使用GPU 0
            task_list=task_list,
            sample_num=sample_num,
            BSZ=BSZ,
            output_dir=output_dir
        )
        
        
