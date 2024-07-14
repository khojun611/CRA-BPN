#!/bin/bash
conda activate TextBPN \
cd CRAFT-pytorch \
python test.py \ 
cd .. \
cd TextBPN \
sh eval.sh
