CUDA_LAUNCH_BLOCKING=1 python3.8 -m torch.distributed.run \
--nnodes= \
--nproc_per_node= \
../train_vit.py \
--kwargs_path "" \
--data_dir "" \
--tb_log_dir "" \
--ckpt_dir "" \
--train_batch_size \
--valid_batch_size \
--epochs
