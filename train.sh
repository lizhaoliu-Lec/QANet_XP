#!/usr/bin/env bash

python cli.py --train --batch_size 12 --learning_rate 1e-3 --optim adam --decay 0.9999 --weight_decay 1e-5 --max_norm_grad 10.0 --dropout 0.0 --head_size 1 --hidden_size 64 --epochs 5 --gpu 0