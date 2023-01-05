#!/bin/bash


# # for hospital-based split
#  /codes/data      /scratch/hokarami/data            C:/DATA/data/processed
PRE="/scratch/hokarami/data"

#p12_full_seft,  p12_full_hosp,                  physio2019_1d_HP_std_AB,          physio2019_1d_HP_std_rand

# declare -i REP=3
DA__label="-event_enc 0 -state -demo       -mod none    -next_mark 0  -sample_label 1 "
TEDA__shpmarklabel="-event_enc 1 -state -demo       -mod single    -next_mark 1  -sample_label 1"



DATA_NAME=" physio2019_1d_HP_std_AB"  
COMMON="-data  $PRE/$DATA_NAME/ -epoch 30 -per 100 -w_pos -batch_size 8  -lr 0.00245  -ES_pat 10 -wandb"

python optuna1.py $COMMON $DA__label -user_prefix "[opt]DA__label-"&
python optuna1.py $COMMON $TEDA__shpmarklabel -user_prefix "[opt]TEDA__shpmarklabel-"
# python Main.py $COMMON $DA__label -user_prefix "[]DA__label-"  &
# python Main.py $COMMON $DA__label -user_prefix "[]DA__label-"  &
# python Main.py $COMMON $DA__label -user_prefix "[]DA__label-"  

# DATA_NAME="physio2019_1d_HP_std_AB"  
# COMMON="-data  $PRE/$DATA_NAME/ -epoch 4 -per 100 -w_pos -batch_size 8  -lr 0.00245  -ES_pat 10 -wandb"

# python Main.py $COMMON $DA__label -user_prefix "[w1]DA__label-"  &
# python Main.py $COMMON $DA__label -user_prefix "[w2]DA__label-"  &
# python Main.py $COMMON $DA__label -user_prefix "[w3]DA__label-"  &
# python Main.py $COMMON $DA__label -user_prefix "[w4]DA__label-" 

# python optuna1.py $COMMON $TEDA__shpmarklabel -user_prefix "[q]TEDA__shpmarklabel-" 










# DA__label="-event_enc 0 -state       -mod none    -next_mark 0  -sample_label 1 "
# TEDA__shpmarklabel="-event_enc 1 -state       -mod single    -next_mark 1  -sample_label 1"



# DATA_NAME="physio2019_1d_HP_std_rand"  
# COMMON="-data  $PRE/$DATA_NAME/ -epoch 30 -per 100 -w_pos -batch_size 8  -lr 0.00245 -cuda -ES_pat 10"

# python optuna1.py $COMMON $DA__label -user_prefix "[q]DA__label-" &
# python optuna1.py $COMMON $TEDA__shpmarklabel -user_prefix "[q]TEDA__shpmarklabel-" 



# # python Main.py $COMMON $DA__label -user_prefix "[t1wd0-sgd]DA__label-"

# python Main.py $COMMON $DA__label -user_prefix "[r1]DA__label-" &
# python Main.py $COMMON $DA__label -user_prefix "[r2]DA__label-" &
# python Main.py $COMMON $DA__label -user_prefix "[r3]DA__label-" &
# python Main.py $COMMON $DA__label -user_prefix "[r4]DA__label-"

# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r1]TEDA__shpmarklabel-" &
# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r2]TEDA__shpmarklabel-" &
# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r3]TEDA__shpmarklabel-" &
# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r4]TEDA__shpmarklabel-"


# DA__label="-event_enc 0 -state -demo       -mod none    -next_mark 0  -sample_label 1 "
# TEDA__shpmarklabel="-event_enc 1 -state -demo       -mod single    -next_mark 1  -sample_label 1"

# PRE="/codes/data"
# DATA_NAME="physio2019_1d_HP_std_rand"  
# COMMON="-data  $PRE/$DATA_NAME/ -epoch 30 -per 100 -w_pos -batch_size 8  -lr 0.00245 -cuda -wandb -ES_pat 30"


# python Main.py $COMMON $DA__label -user_prefix "[r1]DA__label-" &
# python Main.py $COMMON $DA__label -user_prefix "[r2]DA__label-" &
# python Main.py $COMMON $DA__label -user_prefix "[r3]DA__label-" &
# python Main.py $COMMON $DA__label -user_prefix "[r4]DA__label-"

# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r1]TEDA__shpmarklabel-" &
# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r2]TEDA__shpmarklabel-" &
# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r3]TEDA__shpmarklabel-" &
# python Main.py $COMMON $TEDA__shpmarklabel -user_prefix "[r4]TEDA__shpmarklabel-"

# for i in $(eval echo {1..$REP}); do
#     echo python $DA__label -user_prefix "[r$i]DA__label-"
# done

# echo python $DA__label

