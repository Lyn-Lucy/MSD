import argparse
import copy

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

bigname=args.model
image_data_path=args.image_data_path
json_data_path=args.json_data_path

import os
# args.gpu_index.append(args.gpu_index[0] + 4) 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init

from PIL import Image
import requests
from io import BytesIO

def my_process_images(images, image_processor, model_cfg):
    image_aspect_ratio = model_cfg.get("image_aspect_ratio", "original")
    new_images = []
    if image_aspect_ratio == "original":
        # Process images while keeping their original size
        for image in images:
            # Explicitly disable resizing, center cropping, and enforce keeping original dimensions
            image_tensor = image_processor.preprocess(
                image, 
                do_resize=False, 
                do_center_crop=False, 
                return_tensors='pt'
            )['pixel_values'][0]
            new_images.append(image_tensor)
    elif image_aspect_ratio == "448":

        target_size = {"height": 448, "width": 448}
        for image in images:
            image_tensor = image_processor.preprocess(
                image, 
                do_resize=True,  
                size=target_size,  
                do_center_crop=False,
                return_tensors='pt'
            )['pixel_values'][0]
            new_images.append(image_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

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
    num_proc = 1

    def preprocess_function(examples):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "loss_mask": [],
            "image": [],
            "image_size": []
        }
        for i in range(len(examples['id'])):
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            conv.system = ""
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source = examples['conversations'][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                conv.append_message(role, sentence["value"])
            conversation = conv.get_prompt()
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            image_tensor = None
            image_size = None
            if "image" in examples and examples["image"][i] is not None:
                image_path = os.path.join(image_data_path, examples["image"][i])
                image = load_image(image_path)
                image_size = image.size
                image_tensor = process_images([image], image_processor, {"image_aspect_ratio": "pad"})

            input_ids = tokenizer_image_token(
                    conversation, 
                    tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    return_tensors='pt'
            )


            if -200 in input_ids:
                loss_mask = torch.ones(input_ids.shape[0] + 575, dtype=input_ids.dtype)
                cur_len = 1 + 575
            else:
                loss_mask = torch.ones_like(input_ids)
                cur_len = 1

            sep = conv.sep + conv.roles[1] + ": "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())
            turns = conversation.split(conv.sep2)

            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                # cur_len += 2
                if i != 0 and not tokenizer.legacy:
                    cur_len -= 1
                # tokenizer.decode(input_ids[loss_mask[-input_ids.shape[0]:]==1])
            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["image"].append(image_tensor)
            new_examples["image_size"].append(image_size)

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    return ds1



from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto'}
big_model_model_name = get_model_name_from_path(bigname)
bigtokenizer, bigmodel, image_processor, _ = load_pretrained_model(bigname, None, big_model_model_name, **kwargs)
ds = build_dataset_rank(bigtokenizer)
print(ds)
bigmodel.eval()










@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"].cuda()
    image = None
    image_size = None
    if data["image"] != None:
        image = data["image"].to(dtype=torch.float16).cuda()
        image_size = data["image_size"].cuda()
        # import math
        # print("image_token_num",math.floor(image_size[0]/14)*math.floor(image_size[1]/14))
        # print(image.shape)
    # import pdb
    # pdb.set_trace()
    inputs_embeds,_ = bigmodel.get_inputs_embeds(input_ids, image, image_size)

    outs_big = bigmodel(inputs_embeds = inputs_embeds, output_hidden_states=True)

    # outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp = probs[0].max(dim=1).values


    td={"input_ids":input_ids.cpu()[0],"inputs_embeds":inputs_embeds.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}

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

for id,data in tqdm(enumerate(ds), total=len(ds), unit="samples"):
    # if id % 100==0:
    #     print(id,end="\t")
    # if id % 1000 == 0:
    #     print("")
    outdata = ge(data)
    writedata(outdir,outdata)