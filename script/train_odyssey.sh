export CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7"
python -m torch.distributed.launch --nproc_per_node 8 run_train.py \
--exp_name train_odyssey \
--stage odyssey \
--image_size 384 512 \
--seq_len 24 \
--track_point_num 256 \
--batch_size 8 \
--lr 0.0002 \
--wdecay 0.00001 \
--step_max 200000 \
--log_train 100 \
