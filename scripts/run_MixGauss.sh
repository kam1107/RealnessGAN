#!/bin/sh

export CUDA_VISIBLE_DEVICES=0;

SEED=1;

NAME="RealnessGAN-MixGauss-"$SEED;
LOG_NAME="./LOG/log_"$NAME".txt";
if [ ! -f  $LOG_NAME ]; then
	OMP_NUM_THREADS=1 python3 train_toy.py --lr_D .0001 --lr_G .0001 --beta1 .80 --beta2 .999 \
	--gen_every 100 --print_every 50 --G_h_size 400 --D_h_size 200 --gen_extra_images 10000 \
	--input_folder 'data/MixtureGaussian3By3.pk' \
	--total_iters 500 \
	--seed $SEED \
	--G_updates  3 \
	--D_updates  1 \
	--effective_batch_size   100 \
	--batch_size   100 \
	--n_gpu         1  \
	--positive_skew   1.0 \
	--negative_skew  -1.0 \
	--num_outcomes     5 \
	--use_adaptive_reparam   False \
	--relativisticG		True \
	--extra_folder  "./OUTPUT/"$NAME"_Extra" \
	2>&1 | tee $LOG_NAME;
else
	echo "Danger close. The log file at -- $LOG_NAME -- exists!"
fi
