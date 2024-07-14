#!/bin/bash
cd ../
CUDA_LAUNCH_BLOCKING=1 python train_textBPN.py --exp_name Ctw1500 --net resnet50 --max_epoch 1000 --batch_size 12 --gpu 0 --input_size 640 --viz --viz_freq 10 --optim Adam --lr 0.001 --num_workers 24 --save_dir ./model/new_model_aug_rnn 
#--resume model/Synthtext/TextBPN_resnet50_0.pth 
# --viz --viz_freq 80
#--start_epoch 300
# --resume model/Ctw1500/TextBPN_resnet50_560.pth
# --resume model/Ctw1500/TextBPN_resnet50_560.pth
# --resume model/Ctw1500/TextBPN_resnet50_101.pth
