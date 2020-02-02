#!/bin/sh

export CUDA_VISIBLE_DEVICES=0;

SEED=1;

NAME="RealnessGAN-CelebA256-"$SEED;
LOG_NAME="./LOG/log_"$NAME".txt";
if [ ! -f  $LOG_NAME ]; then
	OMP_NUM_THREADS=1 python3 train.py --lr_D .0002 --lr_G .0002 --beta1 .50 --beta2 .999 \
	--gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 5000 \
	--input_folder 'path/to/dataset' \
	--image_size 256 \
	--total_iters 520000 \
	--seed $SEED \
	--num_workers 8 \
	--G_updates  1 \
	--D_updates  1 \
	--effective_batch_size   32 \
	--batch_size   32 \
	--n_gpu         1  \
	--positive_skew   1.0 \
	--negative_skew  -1.0 \
	--num_outcomes     51 \
	--use_adaptive_reparam   True \
	--relativisticG		True \
	--output_folder "./OUTPUT/"$NAME \
	--extra_folder  "./OUTPUT/"$NAME"_Extra" \
	2>&1 | tee $LOG_NAME;
else
	echo "Danger close. The log file at -- $LOG_NAME -- exists!"
fi
