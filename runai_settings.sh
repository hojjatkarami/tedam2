#!/bin/bash

waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}


N_JOBS=5
USER_PREFIX=V126
# p12     -lr 0.001 -weight_decay 0.001  
# p19     -lr 0.001 -weight_decay 1    #DA__label   TE__shpmark

# # for hospital-based split
#  /codes/data      /scratch/hokarami/data             C:/DATA/data/processed

PRE="/scratch/hokarami/data_tedam"
# PRE="C:/DATA/data/processed"

#p12_full_seft,  p12_full_hosp,                  physio2019_1d_HP_std_AB,          physio2019_1d_HP_std_rand
# EXP="  -setting mc2 -test_center 0 " 
# declare -i REP=3


DA__label="-event_enc 0 -state       -mod none    -next_mark 0  -sample_label 1 "
TEDA__shpmarklabel="-event_enc 1 -state       -mod single    -next_mark 1  -sample_label 1"
TEDA__label="-event_enc 1 -state       -mod none    -next_mark 0  -sample_label 1"

# without label
TE__shpmark="-event_enc 1          -mod single    -next_mark 1  -sample_label 0"
TEDA__shpmark="-event_enc 1    -state       -mod single    -next_mark 1  -sample_label 0"
TEDA__shp="-event_enc 1 -state       -mod single    -next_mark 0  -sample_label 0"
TEDA__ml="-event_enc 1 -state       -mod ml    -next_mark 0  -sample_label 0"




DATA_NAME="p12"
COMMON="  -demo  -epoch 30 -per 100 -w_pos -batch_size 128  -lr 0.01 -weight_decay 0.1  -ES_pat 100 -wandb   -w_pos_label 0.7  "
COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"
 


SETTING=" -data  $PRE/$DATA_NAME/ -setting seft  " 

# python main.py  $COEFS $SETTING $COMMON $DA__label -user_prefix "[$USER_PREFIX]" &

COMMON="  -demo  -epoch 30 -per 100 -w_pos -batch_size 128  -lr 0.01 -weight_decay 0.1  -ES_pat 100 -wandb   -w_pos_label 0.7  "
python main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &
python main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &
python main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &
python main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &
python main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &


for i_hosp in {0..0}
do
    for i_split in {0..0}
    do

    SETTING=" -data  $PRE/$DATA_NAME/ -setting seft -test_center $i_hosp -split $i_split " 
    
    # echo $SETTING



    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]"  &


    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TEDA__shpmark -user_prefix "[$USER_PREFIX]" &


    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TEDA__ml -user_prefix "[$USER_PREFIX]" &


    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $DA__label -user_prefix "[$USER_PREFIX]" &

    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &

    done
    
done





DATA_NAME="p19"
COMMON="  -demo  -epoch 100 -per 100 -w_pos -batch_size 128  -lr 0.00245 -weight_decay 0.1  -ES_pat 100 -wandb"
COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"
 

for i_hosp in {0..0}
do
    for i_split in {0..0}
    do

    SETTING=" -data  $PRE/$DATA_NAME/ -setting mc1 -test_center $i_hosp -split $i_split " 
    
    # echo $SETTING



    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]"  &


    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TEDA__shpmark -user_prefix "[$USER_PREFIX]" &


    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TEDA__ml -user_prefix "[$USER_PREFIX]" &


    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $DA__label -user_prefix "[$USER_PREFIX]" &

    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &

    done
    
done


