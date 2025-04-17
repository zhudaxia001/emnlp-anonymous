from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import argparse

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


# 1. 图像像素  提速 
parser = argparse.ArgumentParser(description="Your script description here.")
    
    # 添加参数
parser.add_argument('--sample_num', type=int, default=1000,   # json内 选取多少个样本测 全测1000 实验10
                    help='Number of samples to use (default: 10)')

parser.add_argument('--BSZ', type=int, default=32,
                    help='Batch size (default: 32)')


parser.add_argument('--gpu', type=int, default=0,
                    help='Batch size (default: 32)')

parser.add_argument('--test', type=bool, default=False,
                    help='Batch size (default: 32)')

parser.add_argument('--steps', type=int, default=1000,
                    help='Steps interval for saving model (default: 100)')


parser.add_argument('--output_dir', type=str, default='/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL_math_resize/Stage3_Open/Stage3_Qwen2.5-VL-3B-GRPO-3_tasks_3000/',
                    help='Path(s) to save/load the model (default: /mnt/tenant-home_speed/dhl/VLM-R1-main/Output/Qwen2.5-VL-3B_VQA_det_math_dhl/)')


parser.add_argument('--checkpoint_dir', type=str, default='/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL_math_resize/Stage3_Open/Stage3_Qwen2.5-VL-3B-GRPO-3_tasks_3000/',
                    help='Path(s) to save/load the model (default: /mnt/tenant-home_speed/dhl/VLM-R1-main/Output/Qwen2.5-VL-3B_VQA_det_math_dhl/)')

args = parser.parse_args()

 
steps=args.steps
output_dir=args.output_dir

MODEL_PATH=f"{args.checkpoint_dir}/checkpoint-{args.steps}" 



# MODEL_PATH='/mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_3_coco_iou_fix_reward/checkpoint-1100/'

OUTPUT_PATH="{OUTPUT_DIR}/{STEPS}.json"
BSZ=60
DATA_ROOT = "/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/rec_jsons_processed"
TEST_DATASETS = ['refgta_subsample_resize']



IMAGE_ROOT = "/mnt/tenant-home_speed/dhl/RL_VL3B/test_data/refgta/"

random.seed(42)

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=f"cuda:{args.gpu}",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 设置 tokenizer 的 padding_side
processor.tokenizer.padding_side = 'left'

def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            x1, y1, x2, y2 = bbox
            return bbox, False
    return [0, 0, 0, 0], False


def iou(box1, box2):
    print('iou',box1,box2)
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
        return None
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

# sample_num = 500

for ds in TEST_DATASETS:
    print(f"Processing {ds}...")
    ds_path = '/mnt/tenant-home_speed/dhl/VLM-R1-main/data/rec_jsons_processed/Refgta/refgta_subsample_resize.json'
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    data = data[:args.sample_num]
    messages = []

    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image']).replace('final','final_resize',1)
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
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
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
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(f"cuda:{args.gpu}")

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
        ground_truth = input_example['solution']
        ground_truth_normalized = input_example['normalized_solution']
        model_answer, normalized = extract_bbox_answer(original_output)
        
        # Count correct answers
        correct = 0
        
        model_answer = process_answer(original_output,'detection')
        bbox=model_answer
        if (bbox==[0,0,0,0]) or (bbox is None):
            bbox,_=extract_bbox_answer(original_output)
        
        ground_truth=input_example.get('solution',input_example.get('normalized_solution'))  # refgta的solution不包含<answer> 所以不能process solution
        # print('answer处理前',original_output)
        print(f'bbox answer处理后 {bbox}')
        # print(f'ground_truth {ground_truth}')
       
        ground_truth_normalized = input_example.get("normalized_solution")
        # print('ground_truth_normalized',ground_truth_normalized)
        print('resize_solution',input_example["resize_solution"])
        
        if model_answer is not None:
            if iou(bbox, ground_truth) > 0.5:
                correct = 1
            elif iou(bbox, ground_truth_normalized) > 0.5:
                correct = 1
            elif iou(bbox, input_example["resize_solution"]) > 0.49:
                correct = 1
        correct_number += correct
        
        print('model_answer',model_answer)
        print('ground_truth',ground_truth)
        print('iou',iou(bbox, input_example["resize_solution"]),'correct',correct)
        
        # Create a result dictionary for this example
        result = {
            'question': input_example['problem'],
            'ground_truth': input_example["resize_solution"],
            'model_output': original_output,
            'extracted_answer': model_answer,
            'correct': correct
        }
        final_output.append(result)

    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print('correct_number',correct_number)
    print('数据长度',len(data) )
    print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

    # Save results to a JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    # 会创建 /path/to/your/output/ 目录
    output_path = OUTPUT_PATH.format(OUTPUT_DIR=args.output_dir, STEPS=args.steps)
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print("-"*100)





