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
import math
from PIL import Image
# TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
# IMAGE_ROOT = "/data/shz/dataset/coco"
# TEST_DATASETS = ['refgta_subsample']
# IMAGE_ROOT = "/data/shz/dataset/refgta"

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

parser.add_argument('--task_list', nargs='+', default=['detection','math','classify'],
                    help='List of tasks to perform, e.g., detection classify') # 选择你要测试哪些任务 可以测试多种任务 自动遍历域内和域外数据集测试

parser.add_argument('--model_path', type=str, default='/mnt/tenant-home_speed/dhl/VLM-R1-main/Fix_reward_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou',
                    help='Path(s) to save/load the model (default: /mnt/tenant-home_speed/dhl/VLM-R1-main/Output/Qwen2.5-VL-3B_VQA_det_math_dhl/)') # 这是文件夹 自动测试文件夹下所有的模型

parser.add_argument('--output_dir', type=str, default='/mnt/tenant-home_speed/dhl/VLM-R1-main/Test/3B_CurRL_SFT_Output',
                    help='Path(s) to save/load the model (default: /mnt/tenant-home_speed/dhl/VLM-R1-main/Output/Qwen2.5-VL-3B_VQA_det_math_dhl/)')


args = parser.parse_args()
#-------------------------参数初始化 开始


# 给 任务类型 和 是否跨域 返回 测试json路径：
def get_ds_path(task_type, in_domain=True):
    indomain_dict={
        'detection':"/grpo_sft_data/grpo_data/train/open/coco_3k_2task/detection_coco_test.jsonl",
        'classify':"/grpo_sft_data/grpo_data/train/open/coco_3k_2task/classify_v2_coco_test.jsonl",
        'math':"/grpo_sft_data/grpo_data/train/open/openr1_8k/math_math_test.jsonl"
    } 
    outdomain_dict={ 
        'detection':"",  # /grpo_sft_data/grpo_data/test/Refgta/refgta_subsample_resize.json (test in another python file)
        'classify':"/grpo_data/test/pascal/classify_pascal_voc_test.jsonl",
        'math':"/Curr_REFT/eval/superclevr_test200_counting_problems.jsonl"
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

def run_test_model(model,processor,model_path,gpu_id,task_list, sample_num, BSZ,output_dir):
    with torch.no_grad():
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
                os.makedirs(output_dir, exist_ok=True)
                json_output_path=os.path.join(output_dir,item_add,str(model_path.split('/')[-1].split('-')[-1])+'.json')
                
                if os.path.exists(json_output_path): #说明已经测过 跳过这个checkpoint
                    continue
                    print('跳过测评')
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
                
                    # Preparation for inference
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
                                answer = parse(original_output)
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
                                    
                                    if get_final_answer(original_output)==get_final_answer(input_example['solution']): # 做个选择题匹配
                                        correct = 1.0
                                        if get_final_answer(original_output)!=None:
                                            print('choice sucesuss!!!')
                                            print('math choice:  ','student_answer',get_final_answer(original_output),'ground_truth',get_final_answer(input_example['solution']))
                                # 如果 答案出现在回答中也算对 
                                # Extract answer from solution if it has think/answer tags
                                ground_truth = extract_answer_content(input_example['solution'])
                                student_answer = extract_answer_content(original_output)
                                
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
                                # except Exception:
                                #     print('rule-match failed......')
                                #     pass  # Keep reward as 0.0 if both methods fail
                             
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
                print(f"\nAccuracy of {json_id} 测试--task_type:{task_type} : {accuracy:.2f}%")
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

                # 保存总表
                # print(f"Appending accuracy {accuracy} for checkpoint {json_id} under key {item_add}")
                
                acc_dict[int(json_id)][0][item_add].append(accuracy)
              
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



import threading

model_load_semaphore = threading.Semaphore(8)

def load_model_with_semaphore(j):
    with model_load_semaphore:
        return load_model(j)

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
    
    # 这里会获取dir下的所有checkpoints 然后一起测试
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
    
    # 也可以手动写 测试哪些模型
    # checkpoint_list = \
    #     ['/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-1900',  
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-2000',  
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-2100',
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-2200',
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-2300',
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-2400',
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-2500',
    #     '/mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/Reject_sft_Output/v0-20250301-214307/checkpoint-2600']
                
    print('checkpoint_list', checkpoint_list) # type: ignore
    # 加两个 如果每隔100 steps跑一次
    # 如果已经有这个json文件就跳过 本次测试

  
    # start = 4
    multiprocessing.set_start_method('spawn',force=True)
    dict_list = [{'gpu_id':i%8,'model_path':model_path} for i,model_path in enumerate(checkpoint_list)]
    Max_workers=8

    lock = Lock()
    async_results = []
    print(f'total file check num:{len(dict_list)}')
    # multiprocessing.set_start_method('spawn', force=True)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=Max_workers) as executor:
    
    model_load_semaphore = threading.Semaphore(Max_workers)

    import concurrent.futures
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
                    result = future.result()
                    all_ready_ok_gpu.append(result['gpu_id'])
                    print(f"Task completed with result: {result}")
                    
            
            # 提交任务
            async_result = executor.submit(run_test_model, model, processor, model_path, gpu_id, task_list, sample_num, BSZ,args.output_dir)
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
    
    output_dir =args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    excel_output_dir=output_dir
    # save成excel  保存代码有错

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






