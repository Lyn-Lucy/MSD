import argparse
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--image_data_path', type=str, default=None)
parser.add_argument('--json_data_path', type=str, default=None)

args = parser.parse_args()

modelname=args.model
image_data_path=args.image_data_path
json_data_path=args.json_data_path

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from PIL import Image
import base64
import requests
import numpy as np
from io import BytesIO

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('json', data_files=json_data_path)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 1

    # 处理视觉信息
    def convert_image_to_base64(image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        return f"data:image/jpeg;base64,{base64_bytes.decode('utf-8')}"

    def preprocess_function(examples):
        new_examples = {
            "input_ids": [], # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
            "loss_mask": [],
            "full_chat_text": [],
            "visuals": []
        }

        for i in range(len(examples['id'])):
            full_chat_text = ""
            full_chat_text += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            visuals = []
            source = examples['conversations'][i]
            if source[0]["from"] != "human":
                source = source[1:]
            for j, sentence in enumerate(source):
                role = "user" if sentence["from"] == "human" else "assistant"
                full_chat_text += f"<|im_start|>{role}\n"
                if "<image>" in sentence["value"]:
                    if "image" in examples and examples["image"][i] is not None:
                        image_path = os.path.join(image_data_path, examples["image"][i])
                        image = Image.open(image_path).convert("RGB")
                        height, width = image.size
                        max_size = max(height, width)
                        max_pixels = 1638400
                        if max_size > np.sqrt(max_pixels):
                            scale = np.sqrt(max_pixels) / max_size
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            image = image.resize((new_width, new_height), Image.LANCZOS)
                        
                        visuals.append(image)
                        full_chat_text += "<|vision_start|><|image_pad|><|vision_end|>"
                    
                    content = sentence["value"].replace("<image>", "").strip()
                    full_chat_text += content
                else:
                    full_chat_text += sentence["value"]
                full_chat_text += f"<|im_end|>\n"

            inputs = processor(
                text=full_chat_text, 
                images=visuals if visuals else None, 
                return_tensors="pt", 
                padding=True, 
                max_length=2048, 
                truncation=True
            )

            input_ids = inputs.input_ids[0]
            loss_mask = torch.zeros_like(input_ids)

            assistant_start_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            assistant_end_tokens = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

            assistant_ranges = []
            for i in range(len(input_ids) - len(assistant_start_tokens) + 1):
                if torch.equal(input_ids[i:i+len(assistant_start_tokens)], torch.tensor(assistant_start_tokens)):

                    for j in range(i + len(assistant_start_tokens), len(input_ids) - len(assistant_end_tokens) + 1):
                        if torch.equal(input_ids[j:j+len(assistant_end_tokens)], torch.tensor(assistant_end_tokens)):
                            assistant_ranges.append((i + len(assistant_start_tokens), j))
                            break


            for start, end in assistant_ranges:
                loss_mask[start:end+1] = 1

            new_examples["input_ids"].append(input_ids)
            new_examples["loss_mask"].append(loss_mask)
            new_examples["full_chat_text"].append(full_chat_text)
            new_examples["visuals"].append(visuals[0] if visuals else None)
        
        return new_examples


    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        #num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")
    return ds1




model = Qwen2VLForConditionalGeneration.from_pretrained(modelname, device_map="auto", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained(modelname, max_pixels=12845056, min_pixels=3136)
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)

dataset = build_dataset_rank(tokenizer)

import torch
from tqdm import tqdm

@torch.no_grad()
def get_input_embeds_qwen2vl(input_ids, pixel_values, image_grid_thw, model):
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(model.visual.get_dtype())
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds

class get_last_embed:
    last_embed=None

@torch.no_grad()
def ge(data):

    full_chat_text = data["full_chat_text"]
    visuals = data["visuals"]
    inputs = processor(
        text=full_chat_text, 
        images=[visuals] if visuals is not None else None, 
        return_tensors="pt", 
        padding=True, 
        max_length=2048, 
        truncation=True
    )
    input_ids = inputs.input_ids
    if "pixel_values" in inputs:
        pixel_values = inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw
        input_embeds = get_input_embeds_qwen2vl(input_ids.cuda(), pixel_values.cuda(), image_grid_thw.cuda(), model)
    else:
        input_embeds = get_input_embeds_qwen2vl(input_ids.cuda(), None, None, model)
    
    outs_big = model(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    probs = torch.softmax(outs_big.logits, dim=-1)
    td={"input_ids":input_ids.cpu()[0],"inputs_embeds":input_embeds.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for id, data in enumerate(tqdm(dataset, desc="Processing dataset")):
    outdata = ge(data)
    writedata(outdir, outdata)


