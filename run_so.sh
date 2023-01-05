#!/bin/bash

python Main.py -data C:/DATA/data/processed/data_so/fold1/ -data_label multiclass -per 50 -epoch 30 -user_prefix  tCon_  -time_enc concat \
-te_d_mark 8 -te_d_t 8 \
-mod MHP_multiclass -mark_detach 0 -int_dec thp 

python Main.py -data C:/DATA/data/processed/data_so/fold1/ -data_label multiclass -per 50 -epoch 30 -user_prefix  tSum_  -time_enc sum \
-te_d_mark 8 -te_d_t 8 \
-mod MHP_multiclass -mark_detach 0 -int_dec thp 