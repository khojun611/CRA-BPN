#!/bin/bash
conda activate TextBPN \
cd CRAFT-pytorch \
python test.py \ 
cd .. \
cd TextBPN \
cd scripts \
sh train_CTW1500.sh