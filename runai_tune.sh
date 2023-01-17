#!/bin/bash


# # for hospital-based split
#  /codes/data      /scratch/hokarami/data             C:/DATA/data/processed

PRE="/scratch/hokarami/data_tedam"
# PRE="/scratch/hokarami/data"
# PRE="C:/DATA/data/processed"
# PRE="C:/DATA/data/processed"

#p12_full_seft,  p12_full_hosp,                  physio2019_1d_HP_std_AB,          physio2019_1d_HP_std_rand

# declare -i REP=3
DA__label="-event_enc 0 -state       -mod none    -next_mark 0  -sample_label 1 "
TEDA__shpmarklabel="-event_enc 1 -state       -mod single    -next_mark 1  -sample_label 1"
TE__shpmark="-event_enc 1          -mod single    -next_mark 1  -sample_label 0"
TEDA__shpmark="-event_enc 1    -state       -mod single    -next_mark 1  -sample_label 0"
TEDA__label="-event_enc 1 -state       -mod none    -next_mark 0  -sample_label 1"



COMMON="    -epoch 50 -per 100 -w_pos -batch_size 8  -lr 0.00245 -weight_decay 0.1  -ES_pat 20 -wandb"


DATA_NAME="p12" 

# DA__label
EXP=" -data  $PRE/$DATA_NAME/ -setting rand " 
python tune_optuna.py $EXP $COMMON $DA__label -user_prefix "[T1-rand]DA__label-" &


# TE__shpmark
EXP=" -data  $PRE/$DATA_NAME/ -setting rand " 
python tune_optuna.py $EXP $COMMON $TE__shpmark -user_prefix "[T1-rand]TE__shpmark-" &


DATA_NAME="p19" 

# DA__label
EXP=" -data  $PRE/$DATA_NAME/ -setting rand " 
python tune_optuna.py $EXP $COMMON $DA__label -user_prefix "[T1-rand]DA__label-" &


# TE__shpmark
EXP=" -data  $PRE/$DATA_NAME/ -setting rand " 
python tune_optuna.py $EXP $COMMON $TE__shpmark -user_prefix "[T1-rand]TE__shpmark-" &
