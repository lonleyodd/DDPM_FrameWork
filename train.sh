CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --master_port 9995 \
    --nproc_per_node=2 \
    --use_env main.py \