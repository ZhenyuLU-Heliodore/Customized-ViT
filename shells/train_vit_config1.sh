CUDA_LAUNCH_BLOCKING=1 python3.8 -m torch.distributed.run \
--nnodes=1 --nproc_per_node=8 ../train_vit.py \
--kwargs_path "/home/ubuntu/projects/vit-for-cifar100/configs/config1.yml" \
--tb_log_dir "/home/ubuntu/logs/vit/config1" \
--ckpt_dir "/home/ubuntu/ckpt/vit/config1" \
--train_batch_size 128 \
--valid_batch_size 256 \
--epochs 200