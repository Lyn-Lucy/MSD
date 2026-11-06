"""Reproduce the LLaVA-1.6-7B MSD draft-model training run.

This entry point intentionally keeps the historical objective and optimizer
settings while making data ordering, checkpoint filtering, seeds, and output
provenance explicit. It is isolated from the Qwen experiments in
``main_deepspeed.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from torch.utils.data import Dataset

from eagle.model.cnets import Model
from eagle.model.configs import EConfig
from eagle.model.visual_compression import _cluster_means, _nearest_euclidean_centers


IMAGE_TOKEN_ID = 32000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basepath", required=True)
    parser.add_argument("--datapath-v", required=True)
    parser.add_argument("--datapath-t", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cpdir", required=True)
    parser.add_argument("--run-name", default="llava_v16_7b_msd")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--schedule-epochs", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--v-weight", type=float, default=1.0)
    parser.add_argument("--p-weight", type=float, default=0.1)
    parser.add_argument("--decouple", action="store_true")
    parser.add_argument("--kmeans-clusters", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-micro-steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def indexed_weight(basepath: Path, candidates: List[str]) -> torch.Tensor:
    index_path = basepath / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    key = next((candidate for candidate in candidates if candidate in weight_map), None)
    if key is None:
        raise KeyError(f"none of {candidates!r} exists in {index_path}")
    with safe_open(str(basepath / weight_map[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def source_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("data_"):
        raise ValueError(f"unexpected checkpoint name: {path}")
    return int(stem.removeprefix("data_"))


def ordered_checkpoints(root: str, seed: int) -> List[str]:
    paths = list(Path(root).rglob("*.ckpt"))
    by_index: Dict[int, Path] = {}
    for path in paths:
        index = source_index(path)
        if index in by_index:
            raise ValueError(f"duplicate source index {index}: {by_index[index]} and {path}")
        by_index[index] = path
    if not by_index:
        raise RuntimeError(f"no .ckpt files found under {root}")
    expected = set(range(len(by_index)))
    actual = set(by_index)
    if actual != expected:
        missing = sorted(expected - actual)[:20]
        extra = sorted(actual - expected)[:20]
        raise ValueError(f"non-contiguous source indices under {root}; missing={missing}, extra={extra}")
    indices = list(range(len(by_index)))
    random.Random(seed).shuffle(indices)
    return [str(by_index[index]) for index in indices]


def distributed_manifest(root: str, seed: int, manifest_path: Path, rank: int) -> List[str]:
    resolved_root = str(Path(root).resolve())
    if rank == 0 and not manifest_path.exists():
        paths = ordered_checkpoints(resolved_root, seed)
        payload = {"root": resolved_root, "seed": seed, "count": len(paths), "paths": paths}
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(temporary_path, manifest_path)
    dist.barrier()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload["root"] != resolved_root or payload["seed"] != seed:
        raise ValueError(f"manifest metadata does not match requested data: {manifest_path}")
    if payload["count"] != len(payload["paths"]):
        raise ValueError(f"manifest count mismatch: {manifest_path}")
    return payload["paths"]


class UniformHiddenNoise:
    def __init__(self, std: float):
        self.std = std

    def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.std == 0:
            return hidden_state
        scale = self.std * 512.0 / hidden_state.shape[1]
        return hidden_state + (torch.rand_like(hidden_state) - 0.5) * scale


class MSDDataset(Dataset):
    def __init__(
        self,
        visual_paths: List[str],
        text_paths: List[str],
        zero_embedding: torch.Tensor,
        max_length: int,
        schedule_epochs: int,
        noise_std: float,
        defer_shift: bool = False,
    ) -> None:
        self.visual_paths = visual_paths
        self.text_paths = text_paths
        self.zero_embedding = zero_embedding.cpu().to(torch.bfloat16)
        self.max_length = max_length
        self.schedule_epochs = schedule_epochs
        self.epoch_now = 1
        self.visual_ratio = 0.0
        self.noise = UniformHiddenNoise(noise_std)
        self.defer_shift = defer_shift

    def __len__(self) -> int:
        return min(len(self.visual_paths), len(self.text_paths))

    def on_epoch_end(self) -> None:
        self.epoch_now += 1
        if self.epoch_now <= self.schedule_epochs // 2:
            self.visual_ratio = 0.0
        elif self.epoch_now < self.schedule_epochs:
            self.visual_ratio = (self.epoch_now - self.schedule_epochs // 2) / self.schedule_epochs * 2.0
        else:
            self.visual_ratio = 1.0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        paths = self.visual_paths if random.random() < self.visual_ratio else self.text_paths
        data = torch.load(paths[index], map_location="cpu", weights_only=True)
        input_ids = data["input_ids"][: self.max_length].contiguous()
        inputs_embeds = data["inputs_embeds"][: self.max_length].contiguous()
        hidden_state = data["hidden_state"][: self.max_length].contiguous()
        loss_mask = data["loss_mask"][: self.max_length].to(torch.bool).contiguous()
        length = len(input_ids)
        if not (length == len(inputs_embeds) == len(hidden_state) == len(loss_mask)):
            raise ValueError(f"unaligned cached tensors in {paths[index]}")
        if length < 2:
            raise ValueError(f"cache is too short in {paths[index]}")

        if self.defer_shift:
            return {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds.to(torch.bfloat16),
                "hidden_states": hidden_state.to(torch.bfloat16),
                "target": hidden_state.to(torch.bfloat16),
                "loss_mask": loss_mask,
                "attention_mask": torch.ones(length, dtype=torch.bool),
                "image_tokens_num": int((input_ids == IMAGE_TOKEN_ID).sum().item()),
            }

        shifted_ids = torch.cat((input_ids[1:], input_ids.new_zeros(1)))
        shifted_embeds = torch.cat((inputs_embeds[1:], self.zero_embedding), dim=0)
        target = torch.cat((hidden_state[1:], hidden_state.new_zeros((1, hidden_state.shape[-1]))), dim=0)
        loss_mask[-1] = False
        image_tokens_num = int((input_ids == IMAGE_TOKEN_ID).sum().item())

        return {
            "input_ids": shifted_ids,
            "inputs_embeds": shifted_embeds.to(torch.bfloat16),
            "hidden_states": self.noise(hidden_state.to(torch.bfloat16)),
            "target": target.to(torch.bfloat16),
            "loss_mask": loss_mask,
            "attention_mask": torch.ones(length, dtype=torch.bool),
            "image_tokens_num": image_tokens_num,
        }


class MSDCollator:
    @staticmethod
    def pad_1d(tensor: torch.Tensor, length: int) -> torch.Tensor:
        return F.pad(tensor, (0, length - len(tensor)))

    @staticmethod
    def pad_2d(tensor: torch.Tensor, length: int) -> torch.Tensor:
        return F.pad(tensor, (0, 0, 0, length - len(tensor)))

    def __call__(self, features: List[Dict]) -> Dict:
        max_length = max(len(item["input_ids"]) for item in features)
        return {
            "input_ids": torch.stack([self.pad_1d(item["input_ids"], max_length) for item in features]),
            "inputs_embeds": torch.stack([self.pad_2d(item["inputs_embeds"], max_length) for item in features]),
            "hidden_states": torch.stack([self.pad_2d(item["hidden_states"], max_length) for item in features]),
            "target": torch.stack([self.pad_2d(item["target"], max_length) for item in features]),
            "loss_mask": torch.stack([self.pad_1d(item["loss_mask"], max_length) for item in features]),
            "attention_mask": torch.stack([self.pad_1d(item["attention_mask"], max_length) for item in features]),
            "image_tokens_num": [item["image_tokens_num"] for item in features],
        }


def kmeans_compress_and_shift_batch(
    batch: Dict,
    zero_embedding: torch.Tensor,
    clusters: int,
    noise_std: float,
    device: torch.device,
) -> Dict:
    """Match inference-time visual KMeans before building shifted EAGLE targets."""
    if clusters <= 0:
        raise ValueError(f"clusters must be positive, got {clusters}")

    prepared = []
    for batch_index in range(batch["input_ids"].shape[0]):
        length = int(batch["attention_mask"][batch_index].sum().item())
        input_ids = batch["input_ids"][batch_index, :length].to(device, non_blocking=True)
        inputs_embeds = batch["inputs_embeds"][batch_index, :length].to(device, non_blocking=True)
        hidden_state = batch["hidden_states"][batch_index, :length].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"][batch_index, :length].to(device, non_blocking=True).to(torch.bool)

        visual_indices = torch.nonzero(input_ids == IMAGE_TOKEN_ID, as_tuple=False).flatten()
        visual_count = int(visual_indices.numel())
        if visual_count and bool(loss_mask[visual_indices].any()):
            # The HF LLaVA processor expands one image placeholder into many
            # image ids after creating offset mappings.  Older teacher caches
            # therefore store assistant-mask positions in the unexpanded token
            # coordinate system.  Move those positions past the expanded image
            # block before applying KMeans compression.
            first_visual = int(visual_indices[0].item())
            active = torch.nonzero(loss_mask, as_tuple=False).flatten()
            active = torch.where(active > first_visual, active + visual_count - 1, active)
            active = active[active < len(loss_mask)]
            loss_mask = torch.zeros_like(loss_mask)
            loss_mask[active] = True
        if visual_count > clusters:
            text_indices = torch.nonzero(input_ids != IMAGE_TOKEN_ID, as_tuple=False).flatten()
            visual_inputs = inputs_embeds[visual_indices].float()
            visual_hidden = hidden_state[visual_indices].float()
            permutation = torch.randperm(visual_count, device=device)
            centers = visual_inputs[permutation[:clusters]]
            assignments = _nearest_euclidean_centers(visual_inputs, centers)
            counts = torch.bincount(assignments, minlength=clusters).clamp_min(1).to(torch.float32).unsqueeze(1)

            merged_hidden = _cluster_means(visual_hidden, assignments, counts, clusters).to(hidden_state.dtype)
            merged_inputs = _cluster_means(visual_inputs, assignments, counts, clusters).to(inputs_embeds.dtype)
            merged_ids = input_ids.new_full((clusters,), IMAGE_TOKEN_ID)
            merged_loss_mask = torch.zeros(clusters, dtype=torch.bool, device=device)

            sorted_visual = visual_indices.sort().values
            contiguous = bool(
                sorted_visual[-1] - sorted_visual[0] + 1 == sorted_visual.numel()
                and torch.diff(sorted_visual).min() == 1
            )
            if contiguous:
                first_visual = int(sorted_visual[0].item())
                last_visual = int(sorted_visual[-1].item())
                pre_text = text_indices[text_indices < first_visual]
                post_text = text_indices[text_indices > last_visual]
                input_ids = torch.cat((input_ids[pre_text], merged_ids, input_ids[post_text]))
                inputs_embeds = torch.cat((inputs_embeds[pre_text], merged_inputs, inputs_embeds[post_text]))
                hidden_state = torch.cat((hidden_state[pre_text], merged_hidden, hidden_state[post_text]))
                loss_mask = torch.cat((loss_mask[pre_text], merged_loss_mask, loss_mask[post_text]))
            else:
                sorted_text = text_indices.sort().values
                input_ids = torch.cat((input_ids[sorted_text], merged_ids))
                inputs_embeds = torch.cat((inputs_embeds[sorted_text], merged_inputs))
                hidden_state = torch.cat((hidden_state[sorted_text], merged_hidden))
                loss_mask = torch.cat((loss_mask[sorted_text], merged_loss_mask))

        sequence_length = len(input_ids)
        if sequence_length < 2:
            raise ValueError("KMeans-compressed sequence is too short")
        if noise_std:
            scale = noise_std * 512.0 / sequence_length
            noisy_hidden = hidden_state + (torch.rand_like(hidden_state) - 0.5) * scale
        else:
            noisy_hidden = hidden_state

        shifted_ids = torch.cat((input_ids[1:], input_ids.new_zeros(1)))
        shifted_embeds = torch.cat(
            (inputs_embeds[1:], zero_embedding.to(device=device, dtype=inputs_embeds.dtype)), dim=0
        )
        target = torch.cat((hidden_state[1:], hidden_state.new_zeros((1, hidden_state.shape[-1]))), dim=0)
        loss_mask[-1] = False
        prepared.append(
            {
                "input_ids": shifted_ids,
                "inputs_embeds": shifted_embeds,
                "hidden_states": noisy_hidden,
                "target": target,
                "loss_mask": loss_mask,
                "attention_mask": torch.ones(sequence_length, dtype=torch.bool, device=device),
                "image_tokens_num": int((shifted_ids == IMAGE_TOKEN_ID).sum().item()),
            }
        )

    max_length = max(len(item["input_ids"]) for item in prepared)
    collator = MSDCollator()
    return collator(prepared)


def reduced_mean(value: torch.Tensor) -> float:
    value = value.detach().float()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return (value / dist.get_world_size()).item()


def main() -> None:
    args = parse_args()
    if args.kmeans_clusters < 0:
        raise ValueError("--kmeans-clusters must be non-negative")
    if args.kmeans_clusters and not args.decouple:
        raise ValueError("training with KMeans requires --decouple")
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    basepath = Path(args.basepath)
    config_path = Path(args.config)
    cpdir = Path(args.cpdir)
    if rank == 0:
        cpdir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    visual_paths = distributed_manifest(
        args.datapath_v,
        args.data_seed,
        cpdir / f"{args.run_name}_visual_manifest.json",
        rank,
    )
    if Path(args.datapath_t).resolve() == Path(args.datapath_v).resolve():
        text_paths = visual_paths
    else:
        text_paths = distributed_manifest(
            args.datapath_t,
            args.data_seed,
            cpdir / f"{args.run_name}_text_manifest.json",
            rank,
        )
    train_v = visual_paths[: int(len(visual_paths) * args.train_fraction)]
    train_t = text_paths[: int(len(text_paths) * args.train_fraction)]
    if args.limit is not None:
        train_v = train_v[: args.limit]
        train_t = train_t[: args.limit]

    config = EConfig.from_pretrained(str(config_path))
    embed = indexed_weight(
        basepath,
        ["language_model.model.embed_tokens.weight", "model.embed_tokens.weight"],
    )
    model = Model(config, load_emb=False, train_embed=False, decouple=args.decouple)
    model.embed_tokens.weight.data.copy_(embed.to(model.embed_tokens.weight.dtype))
    for parameter in model.embed_tokens.parameters():
        parameter.requires_grad = False
    zero_embedding = model.embed_tokens(torch.tensor(0))[None].detach()

    dataset = MSDDataset(
        train_v,
        train_t,
        zero_embedding,
        args.max_length,
        args.schedule_epochs,
        args.noise_std,
        defer_shift=args.kmeans_clusters > 0,
    )
    collator = MSDCollator()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable_parameters,
        training_data=dataset,
        collate_fn=collator,
    )

    head_weight = indexed_weight(basepath, ["language_model.lm_head.weight", "lm_head.weight"])
    lm_head = nn.Linear(head_weight.shape[1], head_weight.shape[0], bias=False, device=device, dtype=torch.bfloat16)
    lm_head.weight.data.copy_(head_weight.to(device=device, dtype=torch.bfloat16))
    lm_head.requires_grad_(False)
    lm_head.eval()

    metadata = {
        "run_name": args.run_name,
        "basepath": str(basepath.resolve()),
        "datapath_v": str(Path(args.datapath_v).resolve()),
        "datapath_t": str(Path(args.datapath_t).resolve()),
        "config": str(config_path.resolve()),
        "world_size": world_size,
        "train_samples": len(dataset),
        "epochs": args.epochs,
        "schedule_epochs": args.schedule_epochs,
        "max_length": args.max_length,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "noise_std": args.noise_std,
        "v_weight": args.v_weight,
        "p_weight": args.p_weight,
        "decouple": args.decouple,
        "kmeans_clusters": args.kmeans_clusters,
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
    }
    if rank == 0:
        (cpdir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)

    global_micro_step = 0
    started_at = time.time()
    for epoch in range(args.epochs):
        model_engine.train()
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_micro_steps = 0
        epoch_mask_tokens = torch.zeros((), device=device)
        for micro_step, batch in enumerate(train_loader):
            if args.max_micro_steps_per_epoch is not None and micro_step >= args.max_micro_steps_per_epoch:
                break
            if args.kmeans_clusters:
                batch = kmeans_compress_and_shift_batch(
                    batch,
                    zero_embedding,
                    args.kmeans_clusters,
                    args.noise_std,
                    device,
                )
            hidden_states = batch["hidden_states"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            inputs_embeds = batch["inputs_embeds"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["loss_mask"].to(device, non_blocking=True).unsqueeze(-1).to(torch.float32)

            predict = model_engine(
                hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                image_tokens_num=batch["image_tokens_num"],
            )
            with torch.no_grad():
                target_probability = torch.softmax(lm_head(target), dim=-1)
            output_logits = lm_head(predict)
            token_loss = -(target_probability * torch.log_softmax(output_logits, dim=-1)).sum(dim=-1, keepdim=True)
            denominator = mask.sum().clamp_min(1.0)
            probability_loss = (token_loss * mask).sum() / denominator
            hidden_loss = F.smooth_l1_loss(predict.float(), target.float(), reduction="none").mean(dim=-1, keepdim=True)
            hidden_loss = (hidden_loss * mask).sum() / denominator
            loss = args.v_weight * hidden_loss + args.p_weight * probability_loss

            model_engine.backward(loss)
            model_engine.step()
            epoch_loss_sum += loss.detach()
            epoch_mask_tokens += denominator.detach()
            epoch_micro_steps += 1
            global_micro_step += 1

            if rank == 0 and (micro_step + 1) % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} micro_step={micro_step + 1}/{len(train_loader)} "
                    f"global_step={model_engine.global_steps} loss={loss.item():.6f} "
                    f"hidden_loss={hidden_loss.item():.6f} probability_loss={probability_loss.item():.6f} lr={lr:.8g}",
                    flush=True,
                )

        dataset.on_epoch_end()
        average_loss = epoch_loss_sum / max(epoch_micro_steps, 1)
        average_loss_value = reduced_mean(average_loss)
        elapsed = time.time() - started_at
        if rank == 0:
            print(
                f"epoch_complete={epoch} global_step={model_engine.global_steps} "
                f"average_loss={average_loss_value:.6f} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

        if not args.no_save:
            checkpoint_dir = cpdir / f"{args.run_name}_{epoch}"
            model_engine.save_16bit_model(str(checkpoint_dir))
            dist.barrier()
            if rank == 0:
                shutil.copy2(config_path, checkpoint_dir / "config.json")
                state = {
                    **metadata,
                    "epoch": epoch,
                    "global_steps": model_engine.global_steps,
                    "global_micro_steps": global_micro_step,
                    "average_loss": average_loss_value,
                    "elapsed_seconds": elapsed,
                }
                (checkpoint_dir / "trainer_state.json").write_text(
                    json.dumps(state, indent=2) + "\n", encoding="utf-8"
                )
                shutil.copy2(Path(__file__), checkpoint_dir / Path(__file__).name)
                shutil.copy2(Path(args.deepspeed_config), checkpoint_dir / Path(args.deepspeed_config).name)
            dist.barrier()

    if rank == 0:
        print(f"training_complete global_steps={model_engine.global_steps}", flush=True)


if __name__ == "__main__":
    main()
