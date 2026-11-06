import argparse
import deepspeed
import json
from safetensors import safe_open
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from safetensors import safe_open  # 导入 safetensors 库

from eagle.model.cnets import Model
from eagle.model.configs import EConfig, Qwen2VLConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
import random
import torch.nn.functional as F


def get_embed(path):
    try:
        with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            try:
                embed_path = index_json["weight_map"]["language_model.embed_tokens.weight"]
            except:
                embed_path = index_json["weight_map"]["model.embed_tokens.weight"]
        with safe_open(os.path.join(path, embed_path),
                    framework="pt",
                    device="cpu") as f:
            try:
                tensor_slice = f.get_slice("language_model.model.embed_tokens.weight")
            except:
                tensor_slice = f.get_slice("model.embed_tokens.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        try:
            with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            weights = torch.load(os.path.join(path, emb_path))
            tensor = weights["model.embed_tokens.weight"].float()
        except:
            with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["transformer.wte.weight"]
            weights = torch.load(os.path.join(path, emb_path))
            tensor = weights["transformer.wte.weight"].float()
    return tensor


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

def main():

    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--basepath', type=str, default=None)
    parser.add_argument('--tmpdir_v', type=str,
                        required=True, help="visual instruction-tuning dataset")
    parser.add_argument('--tmpdir_t', type=str,
                        required=True, help="text_only instruction-tuning dataset")
    parser.add_argument('--config', type=str, default='llava_v15_7B_config.json')
    parser.add_argument('--cpdir', type=str, default="", help="The path to save the model")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("-debug", "--debug", action="store_true", help="debug mode")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    train_config = {
        # "lr": 2e-4,
        "bs": 2,
        "gradient_accumulation_steps": 2,
        # "datapath": f"{args.tmpdir}",
        "datapath_v": f"{args.tmpdir_v}",
        "datapath_t": f"{args.tmpdir_t}",
        "is_warmup": True,
        "num_epochs": 20,
        "num_warmup_steps": 2000,
        "total_steps": 800000,
        "p_w": 0.1,
        "v_w": 1.0,
        "head_w": 0.1,
        "num_workers": 2,
        "embeding": True,
        "act": "No",
        "data_noise": True,
        "noise": "uniform",
        "mean": 0.0,
        "std": 0.2,
        "residual": "true,norm",
        "max_len": 4096,
        # "config_path": "/data/llx/code/spd_7b/EAGLE/eagle/train/llava_v15_7B_config.json",
        "config_path": f"{args.config}",
        "b1": 0.9,
        "b2": 0.95,
        "grad_clip": 0.5,
    }

    debug = args.debug
    pool_kernel_size = 4

    torch.backends.cuda.matmul.allow_tf32 = True
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    set_seed(0)
    use_bf16 = True
    accelerator = Accelerator(mixed_precision="bf16")

    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    if rank == 0:
        import wandb
        import shutil
        from datetime import datetime  # 1. 导入 datetime

        if args.debug:
            # 2. 定义你的专属名称
            custom_name = "debug"
        else:
            # custom_name = f"llava_v15_7b_ori_pool_{pool_kernel_size}" 
            custom_name = f"qwen25_7b_base" 

        # 4. 组合成一个唯一的 run ID
        run_id = f"{custom_name}"

        run = wandb.init(project="llava_eagle_13b", config=train_config, id=run_id)  # mode 可以切换 offline/online

        from pathlib import Path

        eagle_root = Path(__file__).resolve().parents[1]
        files_to_save = [
            str(Path(__file__).resolve()),
            str(eagle_root / "model" / "cnets.py"),
            str(eagle_root / "train" / "ds_config.json"),
        ]

        for src in files_to_save:
            # 目标路径（wandb 本地 run 目录下的 files/ 子目录）
            dst_dir = os.path.join(run.dir, "files", os.path.dirname(src).split("eagle/")[-1])
            os.makedirs(dst_dir, exist_ok=True)
            
            # 复制文件为硬拷贝
            shutil.copy2(src, dst_dir)
            
            # 保存到 wandb（会上传副本）
            wandb.save(os.path.join(dst_dir, os.path.basename(src)))

    try:
        with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            try:
                head_path = index_json["weight_map"]["language_model.lm_head.weight"]
            except:
                head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(args.basepath, head_path),
                    framework="pt",
                    device="cpu") as f:
            try:
                tensor_slice = f.get_slice("language_model.lm_head.weight")
            except:
                tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"].float()

    head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
    head.weight.data = tensor


    class CustomDataset(Dataset):
        def __init__(self, datapath_v, datapath_t, total_epoch, transform=None):
            self.data_v = datapath_v
            self.data_t = datapath_t
            self.total_epoch = total_epoch
            self.transform = transform

            self.epoch_now = 1
            self.dataset_v_size = len(self.data_v)
            self.dataset_t_size = len(self.data_t)

            self.current_ratio = 0.0

        def update_ratio(self):
            if self.epoch_now <= self.total_epoch // 2:
                self.current_ratio = 0.0
            elif self.total_epoch // 2 < self.epoch_now < self.total_epoch:
                self.current_ratio = (self.epoch_now - self.total_epoch // 2) / self.total_epoch * 2
            else:
                self.current_ratio = 1.0

        def get_data_source(self):
            if random.random() < self.current_ratio:
                return self.data_v
            else:
                return self.data_t
            
        def __len__(self):
            return min(len(self.data_v), len(self.data_t))
        
        def load_data(self, idx, dataset):
            data = torch.load(dataset[idx])
            return data

        def __getitem__(self, index):
            dataset = self.get_data_source()

            data = self.load_data(index, dataset)
            new_data = {}
            hidden_state = data['hidden_state'][None, :]
            input_ids = data['input_ids'][None, :]
            loss_mask = data["loss_mask"][None, :]
            if "inputs_embeds" in data.keys():
                inputs_embeds = data["inputs_embeds"][None, :]
            else:
                inputs_embeds = embedding(input_ids).to(torch.bfloat16)

            image_tokens_num = 0

            if 151655 in input_ids:
                image_tokens_num = (input_ids == 151655).sum().item()
            # ======== 视觉 Token 剪枝部分 ========
            # ==============================batch-kmeans=========================
            # if (input_ids == 151655).any():
            #     visual_mask = (input_ids == 151655)
            #     text_mask = ~visual_mask

            #     # 索引与形状
            #     visual_indices = torch.nonzero(visual_mask.squeeze(), as_tuple=False).squeeze(1)  # [N_vis]
            #     text_indices = torch.nonzero(text_mask.squeeze(), as_tuple=False).squeeze(1)
            #     N_vis = visual_indices.numel()
            #     H = hidden_state.shape[-1]

            #     # N_vis <= 1 则无需处理
            #     if N_vis > 1:
            #         # 取出视觉部分特征
            #         vis_inputs = inputs_embeds[:, visual_indices, :].squeeze(0).to(torch.float32)      # [N_vis, H]
            #         vis_hidden = hidden_state[:, visual_indices, :].squeeze(0).to(torch.float32)      # [N_vis, H]
                    
            #         # ===============================
            #         # 🚀 Mini-batch 近似 KMeans 聚类压缩
            #         # ===============================
            #         K = min(32, N_vis)  # 你可以自由调整保留的视觉token数
            #         perm = torch.randperm(N_vis, device=vis_inputs.device)
            #         centers = vis_inputs[perm[:K]]  # [K, H] 随机初始化聚类中心

            #         dist = torch.cdist(vis_inputs, centers, p=2)   # [N_vis, K]
            #         assignments = dist.argmin(dim=-1)              # [N_vis]

            #         # scatter 向量化求均值聚合
            #         counts = torch.bincount(assignments, minlength=K).clamp(min=1).unsqueeze(1).to(vis_inputs.device)
            #         merged_sum_hidden = torch.zeros(K, H, device=vis_hidden.device)
            #         merged_sum_inputs = torch.zeros(K, H, device=vis_inputs.device)

            #         merged_sum_hidden = merged_sum_hidden.index_add(0, assignments, vis_hidden)
            #         merged_sum_inputs = merged_sum_inputs.index_add(0, assignments, vis_inputs)

            #         merged_hidden = merged_sum_hidden / counts  # [K, H]
            #         merged_inputs = merged_sum_inputs / counts  # [K, H]

            #         # ===============================
            #         # 构造视觉token id 与 loss_mask
            #         # ===============================
            #         vis_ids_multi = torch.full((1, K), 151655, dtype=input_ids.dtype, device=input_ids.device)
            #         vis_loss_val = loss_mask[:, visual_indices[:K]]  # [1, K]

            #         # 判断视觉token是否为连续块（通常是）
            #         vis_sorted = torch.sort(visual_indices).values
            #         is_contiguous = (vis_sorted[-1] - vis_sorted[0] + 1 == vis_sorted.numel()) and \
            #                         (torch.diff(vis_sorted).min() == 1)

            #         if is_contiguous:
            #             first_vis = vis_sorted[0].item()
            #             last_vis  = vis_sorted[-1].item()

            #             pre_text_idx  = text_indices[text_indices < first_vis]
            #             post_text_idx = text_indices[text_indices > last_vis]

            #             # 拼接：前文本 + 聚类视觉 + 后文本
            #             input_ids = torch.cat([
            #                 input_ids[:, pre_text_idx],
            #                 vis_ids_multi,
            #                 input_ids[:, post_text_idx]
            #             ], dim=1)

            #             hidden_state = torch.cat([
            #                 hidden_state[:, pre_text_idx, :],
            #                 merged_hidden.unsqueeze(0).to(hidden_state.dtype),
            #                 hidden_state[:, post_text_idx, :]
            #             ], dim=1)

            #             inputs_embeds = torch.cat([
            #                 inputs_embeds[:, pre_text_idx, :],
            #                 merged_inputs.unsqueeze(0).to(inputs_embeds.dtype),
            #                 inputs_embeds[:, post_text_idx, :]
            #             ], dim=1)

            #             loss_mask = torch.cat([
            #                 loss_mask[:, pre_text_idx],
            #                 vis_loss_val,
            #                 loss_mask[:, post_text_idx]
            #             ], dim=1)
            #         else:
            #             # 若非连续，安全兜底：文本保持原序，视觉token追加到末尾
            #             sorted_text_idx = torch.sort(text_indices).values

            #             input_ids = torch.cat([
            #                 input_ids[:, sorted_text_idx],
            #                 vis_ids_multi
            #             ], dim=1)

            #             hidden_state = torch.cat([
            #                 hidden_state[:, sorted_text_idx, :],
            #                 merged_hidden.unsqueeze(0).to(hidden_state.dtype)
            #             ], dim=1)

            #             inputs_embeds = torch.cat([
            #                 inputs_embeds[:, sorted_text_idx, :],
            #                 merged_inputs.unsqueeze(0).to(inputs_embeds.dtype)
            #             ], dim=1)

            #             loss_mask = torch.cat([
            #                 loss_mask[:, sorted_text_idx],
            #                 vis_loss_val
            #             ], dim=1)

            #     # 更新视觉token数量统计
            #     image_tokens_num = (input_ids == 151655).sum().item()
            # ==============================only text=========================
            # if (input_ids == 151655).any():
            #     # 视觉 token mask
            #     visual_mask = (input_ids[0] == 151655)
            #     text_mask = ~visual_mask  # 只保留非 151655 的位置

            #     # 直接筛选出文本部分
            #     input_ids = input_ids[:, text_mask]
            #     hidden_state = hidden_state[:, text_mask, :]
            #     inputs_embeds = inputs_embeds[:, text_mask, :]
            #     loss_mask = loss_mask[:, text_mask]

            #     # 更新视觉 token 数量（现在为 0）
            #     image_tokens_num = 0

            input_ids = input_ids[:, :train_config["max_len"]]
            inputs_embeds = inputs_embeds[:, :train_config["max_len"], :]
            hidden_state = hidden_state[:, :train_config["max_len"], :]
            loss_mask = loss_mask[:, :train_config["max_len"]]
                

            length = hidden_state.shape[1]
            attention_mask = [1] * length
            loss_mask = loss_mask[0].tolist()
            loss_mask[-1] = 0

            input_ids_target = input_ids[:, 1:]
            zeropadding = torch.tensor([[0]])
            input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

            inputs_embeds_target = inputs_embeds[:, 1:, :]
            inputs_embeds_target = torch.cat((inputs_embeds_target, zeropadding_embed), dim=1)
            new_data["inputs_embeds"] = inputs_embeds_target.to(torch.bfloat16)

            target = hidden_state[:, 1:, :]
            zeropadding = torch.zeros(1, 1, target.shape[2])
            target = torch.cat((target, zeropadding), dim=1)
            loss_mask[-1] = 0
            new_data["attention_mask"] = attention_mask
            new_data["loss_mask"] = loss_mask
            new_data["target"] = target.to(torch.bfloat16)
            new_data["hidden_state_big"] = hidden_state.to(torch.bfloat16)
            new_data["input_ids"] = input_ids_target
            new_data["image_tokens_num"] = image_tokens_num


            if self.transform:
                new_data = self.transform(new_data)

            return new_data
        
        def on_epoch_end(self):
            self.epoch_now += 1
            self.update_ratio()


    class DataCollatorWithPadding:

        def paddingtensor(self, intensors, N):
            B, n, S = intensors.shape
            padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
            outtensors = torch.cat((intensors, padding_tensor), dim=1)
            return outtensors

        def paddingtensor2D(self, intensors, N, have_img=False):
            if have_img == True:
                img_padding_tensor = torch.zeros(intensors.shape[0], 576 // (pool_kernel_size*pool_kernel_size) - 1, dtype=intensors.dtype)
                intensors = torch.cat((img_padding_tensor, intensors), dim=1)
            if intensors.shape[1] >= train_config["max_len"]:
                intensors = intensors[:,:train_config["max_len"]]
            B, n = intensors.shape
            # N maybe 2048

            padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
            outtensors = torch.cat((intensors, padding_tensor), dim=1)
            return outtensors

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            max_length = max(item['hidden_state_big'].shape[1] for item in features)
            batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length, have_img=(-200 in item['input_ids'])) for item in features])
            batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
            batch_inputs_embeds = torch.cat([self.paddingtensor(item['inputs_embeds'], max_length) for item in features])
            batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
            batch_loss_mask = torch.tensor(
                [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
            batch_attention_mask = torch.tensor(
                [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
            batch_image_tokens_num = [item['image_tokens_num'] for item in features]
            # batch_loss_mask = torch.ones_like(batch_loss_mask)
            # batch_attention_mask=torch.ones_like(batch_attention_mask)
            batch = {
                "input_ids": batch_input_ids,
                "inputs_embeds": batch_inputs_embeds,
                "hidden_states": batch_hidden_states,
                "target": batch_target,
                "attention_mask": batch_attention_mask,
                "loss_mask": batch_loss_mask,
                "image_tokens_num": batch_image_tokens_num
            }
            return batch


    def top_accuracy(output, target, topk=(1,)):
        # output.shape (bs, num_classes), target.shape (bs, )
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return res

    def compute_loss(target, target_p, predict, loss_mask):
        out_head = head_engine(predict)
        out_logp = nn.LogSoftmax(dim=2)(out_head)
        plogp = target_p * out_logp
        # ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
        ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)

        # vloss = criterion(predict, target.to(rank))

        vloss = criterion(predict.float(), data["target"].float().to(rank)).to(torch.bfloat16)
        # vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
        vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)

        return vloss, ploss, out_head

    llama_path = args.basepath
    embed_tensor = get_embed(llama_path)
    embedding = torch.nn.Embedding(embed_tensor.shape[0], embed_tensor.shape[1])
    embedding.weight.data = embed_tensor
    for param in embedding.parameters():
        param.requires_grad = False

    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        aug = None

    datapath_v = list_files(train_config["datapath_v"])
    datapath_t = list_files(train_config["datapath_t"])

    if debug == True:
        traindatapath_v = datapath_v[:10]
        traindatapath_t = datapath_t[:10]
        testdatapath_v = datapath_v[100:110]
        testdatapath_t = datapath_t[100:110]
    else:
        traindatapath_v = datapath_v[:int(len(datapath_v) * 0.95)]
        traindatapath_t = datapath_t[:int(len(datapath_t) * 0.95)]
                                    
        testdatapath_v = datapath_v[int(len(datapath_v) * 0.95):]
        testdatapath_t = datapath_t[int(len(datapath_t) * 0.95):]
    traindataset = CustomDataset(traindatapath_v, traindatapath_t, total_epoch=train_config["num_epochs"], transform=aug)
    testdataset = CustomDataset(testdatapath_v, testdatapath_t, total_epoch=train_config["num_epochs"])
    # test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
    #                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

    if rank == 0:
        if not os.path.exists(args.cpdir):
            os.makedirs(args.cpdir)

    if "Qwen" in args.basepath:
        config = Qwen2VLConfig.from_pretrained(train_config["config_path"])
    else:
        config = EConfig.from_pretrained(train_config["config_path"])

    model = Model(config, path=args.basepath, load_emb=True, init=False, train_embed=False, decouple=False)
    zeropadding_embed = model.embed_tokens(torch.tensor(0))[None, None, :].to(dtype=(torch.bfloat16 if use_bf16 else torch.float16)).detach() 

    criterion = nn.SmoothL1Loss(reduction="none")

    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]
    is_warmup = train_config["is_warmup"]

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                    model=model,
                                                                    model_parameters=model.parameters(),
                                                                    training_data=traindataset,
                                                                    collate_fn=DataCollatorWithPadding()
                                                                    )

    head_engine, _, test_loader, _ = deepspeed.initialize(args=args,
                                                        model=head,
                                                        model_parameters=head.parameters(),
                                                        training_data=testdataset,
                                                        collate_fn=DataCollatorWithPadding()
                                                        )


    for param in head.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.train()
        for batch_idx, data in enumerate(tqdm(train_loader)):

            model.zero_grad()
            predict = model_engine(data["hidden_states"].to(rank), input_ids=data["input_ids"].to(rank),
                                inputs_embeds=data["inputs_embeds"].to(rank),
                                attention_mask=data["attention_mask"].to(rank),
                                image_tokens_num=data["image_tokens_num"],
                                )

            with torch.no_grad():
                target_head = head_engine(data["target"].to(rank))
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            loss_mask = data["loss_mask"][:, :, None].to(rank)
            vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)

            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            # loss.backward()
            model_engine.backward(loss)
            # accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])

            model_engine.step()

            with torch.no_grad():
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            if rank == 0 and ct != 0:
                logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                        "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
                for id, i in enumerate(top_3acc):
                    logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
                wandb.log(logdict)
                # for id,i in enumerate(top_3acc):
                #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

            del ploss, vloss
            epoch_loss += loss.item()
            num_batches += 1
        train_loader.dataset.on_epoch_end()
        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        epoch_loss /= num_batches
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
        if accelerator.is_local_main_process:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
            wandb.log({"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss})

        # model_engine.save_16bit_model(f"{args.cpdir}/state_{epoch}")
        # if epoch % 10 == 0:
        #     deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.cpdir}/state_{epoch}")
        if debug == False:
            name = f"qwen25_7b_base0"
            model_engine.save_16bit_model(f"{args.cpdir}/{name}_{epoch}")
            torch.distributed.barrier()
            config = train_config["config_path"]
            import shutil
            target_file = f"{args.cpdir}/{name}_{epoch}/config.json"
            shutil.copy(config, target_file)
        torch.distributed.barrier()

if __name__ == "__main__":
    main()