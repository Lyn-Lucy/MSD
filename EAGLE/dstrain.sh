# !/bin/bash
# export WANDB_API_KEY=84ed4d85665f0a5672c52a41b3f80ff02b3dd3f5
export WANDB_MODE=offline


cd eagle/train

deepspeed --master_port 29504 --include localhost:0 main_deepspeed.py --deepspeed_config ds_config.json \
    --tmpdir_v "/data/llx/code/msd/datas/eagle_data/no_sys_llava15_7b_0_67_mufp16" \
    --tmpdir_t "/data/llx/code/msd/datas/eagle_data/no_sys_llava15_7b_0_67_mufp16" \
    --basepath "/home/llx/models/models--liuhaotian--llava-v1.5-7b" \
    --cpdir "/data/llx/code/msd/datas/ckpt" \
    --config "/data/llx/code/msd/EAGLE/eagle/train/llava_v15_7B_config.json" \
#     --debug



deepspeed --master_port 29504 --include localhost:0 main_deepspeed.py --deepspeed_config ds_config.json \
    --tmpdir_v "/data/llx/code/msd/datas/eagle_data/qwen2vl_0_67" \
    --tmpdir_t "/data/llx/code/msd/datas/eagle_data/qwen2vl_0_67" \
    --basepath "/home/llx/models/models--Qwen--Qwen2-VL-7B-Instruct" \
    --cpdir "/data/llx/code/msd/datas/ckpt/qwen2vl" \
    --config "/data/llx/code/msd/EAGLE/eagle/train/qwen2vl_config.json" \
    --debug \