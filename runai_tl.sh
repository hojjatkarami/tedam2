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



COMMON="    -epoch 50 -per 100 -w_pos -batch_size 8  -lr 0.000245 -weight_decay 0.1  -ES_pat 20 -wandb"


DATA_NAME="p12" 

EXP=" -data  $PRE/$DATA_NAME/ -setting mc1  -test_center 0 -split 0 " 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R4-mc1]TEDA__shpmarklabel-" &

EXP=" -data  $PRE/$DATA_NAME/ -setting mc1  -test_center 0 -split 1 " 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R4-mc1]TEDA__shpmarklabel-" &

EXP=" -data  $PRE/$DATA_NAME/ -setting mc1  -test_center 0 -split 2 " 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R4-mc1]TEDA__shpmarklabel-" &

EXP=" -data  $PRE/$DATA_NAME/ -setting mc1  -test_center 0 -split 3 " 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R4-mc1]TEDA__shpmarklabel-" &

EXP=" -data  $PRE/$DATA_NAME/ -setting mc1  -test_center 0 -split 4 " 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R4-mc1]TEDA__shpmarklabel-" &


# DATA_NAME="p19" 
# EXP=" -data  $PRE/$DATA_NAME/ -setting mc2 -test_center 0 " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-mc2]TEDA__shpmarklabel-" &



DATA_NAME="p12" 
tl_path="/scratch/hokarami/data_tedam/p12-mc2-H0/[R3-mc2]TEDA__shpmarklabel-1182158/"

# EXP=" -data  $PRE/$DATA_NAME/ -setting tl -test_center 0 -split 0 -transfer_learning  $tl_path " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-tl]TEDA__shpmarklabel-" &

# EXP=" -data  $PRE/$DATA_NAME/ -setting tl -test_center 0 -split 1 -transfer_learning  $tl_path " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-tl]TEDA__shpmarklabel-" &

# EXP=" -data  $PRE/$DATA_NAME/ -setting tl -test_center 0 -split 2 -transfer_learning  $tl_path " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-tl]TEDA__shpmarklabel-" &

# EXP=" -data  $PRE/$DATA_NAME/ -setting tl -test_center 0 -split 3 -transfer_learning  $tl_path " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-tl-fr]TEDA__shpmarklabel-" &

# EXP=" -data  $PRE/$DATA_NAME/ -setting tl -test_center 0 -split 4 -transfer_learning  $tl_path " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-tl]TEDA__shpmarklabel-" 



# DATA_NAME="p19" 
# tl_path="/scratch/hokarami/data_tedam/p19-mc2-H0/-mc2]TEDA__shpmarklabel-1187968/"
# EXP=" -data  $PRE/$DATA_NAME/ -setting mc2 -test_center 0 -transfer_learning  $tl_path " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[TL][R3-mc2]TEDA__shpmarklabel-"


# DATA_NAME="p19" 
# EXP=" -data  $PRE/$DATA_NAME/ -setting mc2 -test_center 0 " 
# python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-mc2]TEDA__shpmarklabel-" &

