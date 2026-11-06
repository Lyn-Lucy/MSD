"""Prepare LLaVA-NeXT hidden-state caches on one or more 8-GPU nodes.

The saved ``inputs_embeds`` and ``hidden_state`` tensors are compatible with
``eagle.train.main_deepspeed``.  Node-level ``RANK`` and ``WORLD_SIZE`` follow
DeepSpec's launcher convention; each node spawns one worker per visible GPU.
"""

import argparse
import json
import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GENLONG_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
GENLONG_LENGTH_INSTRUCTION = "Please answer with at least 1000 words."
TRAILING_SEPARATOR_IDS = (13, 29871)

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("teacher", "generate"),
        default="generate",
        help="teacher caches annotated answers; generate caches prompt and generated tokens.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("plain", "genlong"),
        default="plain",
        help="genlong reproduces the original LCS long-answer system and length prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--processes-per-node",
        type=int,
        default=None,
        help="Defaults to all visible GPUs. Primarily useful for smoke tests.",
    )
    args = parser.parse_args()
    if args.start < 0 or (args.end is not None and args.end < args.start):
        parser.error("--start/--end define an invalid range")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    return args


def load_annotations(path: str) -> List[Dict]:
    annotation_path = Path(path)
    if annotation_path.suffix == ".jsonl":
        with annotation_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {annotation_path}, got {type(data).__name__}")
    return data


def role_of(turn: Dict) -> str:
    role = str(turn.get("from", turn.get("role", ""))).lower()
    if role in ("human", "user"):
        return "user"
    if role in ("gpt", "assistant"):
        return "assistant"
    raise ValueError(f"Unsupported conversation role: {role!r}")


def value_of(turn: Dict) -> str:
    value = turn.get("value", turn.get("content", ""))
    if isinstance(value, str):
        return value.strip()
    raise TypeError("Only string conversation content is supported")


def _append(parts: List[str], value: str, position: int) -> int:
    parts.append(value)
    return position + len(value)


def render_conversation(
    sample: Dict,
    *,
    mode: str,
    has_image: bool,
    prompt_style: str = "plain",
) -> Tuple[str, List[Tuple[int, int]]]:
    turns = list(sample.get("conversations", sample.get("messages", [])))
    if not turns:
        raise ValueError("Sample has no conversations/messages")

    if mode == "generate":
        assistant_indices = [i for i, turn in enumerate(turns) if role_of(turn) == "assistant"]
        if assistant_indices:
            turns = turns[: assistant_indices[-1]]

    parts: List[str] = []
    assistant_spans: List[Tuple[int, int]] = []
    position = _append(parts, "<s> ", 0)
    if prompt_style == "genlong":
        position = _append(parts, GENLONG_SYSTEM_PROMPT + " ", position)
    inserted_image = False

    for turn in turns:
        role = role_of(turn)
        value = value_of(turn)
        contains_image = "<image>" in value
        value = value.replace("<image>", "").strip()
        if role == "user":
            position = _append(parts, "USER: ", position)
            if has_image and (contains_image or not inserted_image):
                position = _append(parts, "<image>\n", position)
                inserted_image = True
            position = _append(parts, value, position)
            if prompt_style == "genlong":
                position = _append(parts, " " + GENLONG_LENGTH_INSTRUCTION, position)
            position = _append(parts, " ", position)
        else:
            position = _append(parts, "ASSISTANT: ", position)
            span_start = position
            position = _append(parts, value, position)
            assistant_spans.append((span_start, position))
            position = _append(parts, " ", position)

    if mode == "generate":
        position = _append(parts, "ASSISTANT:", position)

    text = "".join(parts)
    if has_image and "<image>" not in text:
        raise ValueError("Image sample did not produce an <image> token")
    return text, assistant_spans


def resolve_image_path(sample: Dict, image_root: Optional[str]) -> Optional[Path]:
    image_value = sample.get("image")
    if not image_value:
        return None
    image_path = Path(str(image_value))
    if not image_path.is_absolute():
        if image_root is None:
            raise ValueError("--image-root is required for relative image paths")
        image_path = Path(image_root) / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    return image_path


def loss_mask_from_offsets(
    offsets: torch.Tensor,
    assistant_spans: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    mask = torch.zeros(offsets.shape[0], dtype=torch.bool)
    for token_index, (token_start, token_end) in enumerate(offsets.tolist()):
        if token_end <= token_start:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in assistant_spans):
            mask[token_index] = True
    return mask


def loss_mask_from_token_ids(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Build an answer mask after multimodal expansion invalidates text offsets."""
    token_ids = input_ids[0]
    marker = token_ids.new_tensor(
        tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    )
    matches = (token_ids.unfold(0, marker.numel(), 1) == marker).all(dim=1).nonzero().flatten()
    if matches.numel() == 0:
        raise ValueError("Teacher sequence has no ASSISTANT: marker")

    answer_start = int(matches[-1].item()) + marker.numel()
    answer_stop = token_ids.numel()
    while answer_stop > answer_start and int(token_ids[answer_stop - 1].item()) in TRAILING_SEPARATOR_IDS:
        answer_stop -= 1
    if answer_stop <= answer_start:
        raise ValueError("Teacher sequence has an empty final assistant answer")

    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    mask[answer_start:answer_stop] = True
    return mask


def prepare_inputs(processor, text: str, image: Optional[Image.Image], max_length: int):
    return processor(
        images=image,
        text=text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )


def to_device(batch, device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def cache_sample(model, processor, sample: Dict, source_index: int, args, device):
    image_path = resolve_image_path(sample, args.image_root)
    text, assistant_spans = render_conversation(
        sample,
        mode=args.mode,
        has_image=image_path is not None,
        prompt_style=args.prompt_style,
    )
    image = Image.open(image_path).convert("RGB") if image_path is not None else None
    try:
        prepared = prepare_inputs(processor, text, image, args.max_length)
    finally:
        if image is not None:
            image.close()

    offsets = prepared.pop("offset_mapping")[0]
    batch = to_device(prepared, device)
    prompt_length = int(batch["input_ids"].shape[1])

    if args.mode == "teacher":
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        inputs_embeds = outputs.hidden_states[0]
        hidden_state = outputs.hidden_states[-1]
        input_ids = batch["input_ids"]
        if image_path is not None:
            loss_mask = loss_mask_from_token_ids(batch["input_ids"], processor.tokenizer).unsqueeze(0)
        else:
            loss_mask = loss_mask_from_offsets(offsets, assistant_spans).unsqueeze(0).to(device)
        generated_length = 0
    else:
        max_positions = int(getattr(model.config.text_config, "max_position_embeddings", args.max_length))
        available_tokens = max_positions - prompt_length
        if available_tokens < 1:
            raise ValueError(f"Prompt length {prompt_length} leaves no room for generation")
        generation_kwargs = {
            "max_new_tokens": min(args.max_new_tokens, available_tokens),
            "do_sample": args.temperature > 0,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "use_cache": True,
        }
        if args.temperature > 0:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
        outputs = model.generate(**batch, **generation_kwargs)
        if not outputs.hidden_states:
            raise RuntimeError("generate() did not return hidden states")
        inputs_embeds = torch.cat([step[0] for step in outputs.hidden_states], dim=1)
        hidden_state = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
        state_length = int(hidden_state.shape[1])
        input_ids = outputs.sequences[:, :state_length]
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        loss_mask[:, max(prompt_length - 1, 0) :] = True
        generated_length = int(outputs.sequences.shape[1] - prompt_length)

    input_ids = input_ids[0].detach().cpu().contiguous()
    inputs_embeds = inputs_embeds[0].detach().cpu().contiguous()
    hidden_state = hidden_state[0].detach().cpu().contiguous()
    loss_mask = loss_mask[0].detach().cpu().contiguous()
    if not (len(input_ids) == len(inputs_embeds) == len(hidden_state) == len(loss_mask)):
        raise RuntimeError(
            "Cache length mismatch: "
            f"ids={len(input_ids)}, embeds={len(inputs_embeds)}, "
            f"hidden={len(hidden_state)}, loss={len(loss_mask)}"
        )

    image_token_id = int(model.config.image_token_index)
    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "hidden_state": hidden_state,
        "loss_mask": loss_mask,
        "image_mask": input_ids.eq(image_token_id),
        "prompt_length": prompt_length,
        "generated_length": generated_length,
        "source_index": source_index,
        "sample_id": str(sample.get("id", source_index)),
        "mode": args.mode,
    }


def atomic_torch_save(data: Dict, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(data, temporary_path)
    os.replace(temporary_path, path)


def select_indices(num_samples: int, args) -> List[int]:
    indices = list(range(num_samples))
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(indices)
    end = num_samples if args.end is None else min(args.end, num_samples)
    indices = indices[args.start:end]
    if args.limit is not None:
        indices = indices[: args.limit]
    return indices


def worker(local_rank: int, local_world_size: int, args: argparse.Namespace) -> None:
    node_rank = int(os.environ.get("RANK", "0"))
    node_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = node_rank * local_world_size + local_rank
    global_world_size = node_world_size * local_world_size
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + global_rank)

    annotations = load_annotations(args.annotation_path)
    indices = select_indices(len(annotations), args)
    rank_start = len(indices) * global_rank // global_world_size
    rank_end = len(indices) * (global_rank + 1) // global_world_size
    rank_indices = indices[rank_start:rank_end]
    rank_dir = Path(args.output_dir) / f"rank_{global_rank:05d}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    completed = 0
    skipped = 0
    failed = 0
    error_path = rank_dir / "errors.jsonl"
    progress = tqdm(rank_indices, desc=f"rank {global_rank}", dynamic_ncols=True)
    for source_index in progress:
        output_path = rank_dir / f"data_{source_index:08d}.ckpt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            data = cache_sample(model, processor, annotations[source_index], source_index, args, device)
            atomic_torch_save(data, output_path)
            completed += 1
        except Exception as exc:
            failed += 1
            error = {
                "source_index": source_index,
                "sample_id": str(annotations[source_index].get("id", source_index)),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error, ensure_ascii=False) + "\n")
            if args.fail_fast:
                raise
        progress.set_postfix(completed=completed, skipped=skipped, failed=failed)

    status = {
        "global_rank": global_rank,
        "global_world_size": global_world_size,
        "source_range": [rank_start, rank_end],
        "assigned": len(rank_indices),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }
    status_path = rank_dir / "status.json"
    temporary_status_path = status_path.with_suffix(".json.tmp")
    with temporary_status_path.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    os.replace(temporary_status_path, status_path)


def main() -> None:
    args = parse_args()
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError("At least one visible CUDA GPU is required")
    local_world_size = args.processes_per_node or visible_gpus
    if local_world_size < 1 or local_world_size > visible_gpus:
        raise ValueError(
            f"--processes-per-node must be in [1, {visible_gpus}], got {local_world_size}"
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(
        worker,
        args=(local_world_size, args),
        nprocs=local_world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
