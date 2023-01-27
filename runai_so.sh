#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}


N_JOBS=5
USER_PREFIX=SO
# p12     -lr 0.001 -weight_decay 0.001  
# p19     -lr 0.001 -weight_decay 1    #DA__label   TE__shpmark

# # for hospital-based split
#  /codes/data      /scratch/hokarami/data             C:/DATA/data/processed

PRE="/scratch/hokarami/data_tedam"
PRE="C:/DATA/data/processed"
PRE="/scratch/hokarami/new"
PRE="/scratch/hokarami/data_old"

#p12_full_seft,  p12_full_hosp,                  physio2019_1d_HP_std_AB,          physio2019_1d_HP_std_rand
# EXP="  -setting mc2 -test_center 0 "
# declare -i REP=3


DA__label="-event_enc 0 -state       -mod none    -next_mark 0  -sample_label 1 "
TEDA__shpmarklabel="-event_enc 1 -state       -mod single    -next_mark 1  -sample_label 1"
TEDA__label="-event_enc 1 -state       -mod none    -next_mark 0  -sample_label 1"

# without label
TE__shpmark="-event_enc 1          -mod single    -next_mark 1  -sample_label 0"
TE__markmc="-event_enc 1          -mod mc    -next_mark 1  -sample_label 0"
TE__markml="-event_enc 1          -mod ml    -next_mark 1  -sample_label 0"

TEDA__shpmark="-event_enc 1    -state       -mod single    -next_mark 1  -sample_label 0"
TEDA__shp="-event_enc 1 -state       -mod single    -next_mark 0  -sample_label 0"
TEDA__ml="-event_enc 1 -state       -mod ml    -next_mark 0  -sample_label 0"




DATA_NAME="synthea_full"

COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"
 


SETTING=" -data  $PRE/$DATA_NAME/ -split '' -data_label multiclass" 

COMMON="   -epoch 30 -per 100  -batch_size 8  -lr 0.0003 -weight_decay 0.1  -ES_pat 100 -wandb  -time_enc concat "
# python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" &

COMMON="   -epoch 30 -per 100 -batch_size 8  -lr 0.0003 -weight_decay 0.1  -ES_pat 100 -wandb  -time_enc sum "
# python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" &


# COMMON="   -epoch 30 -per 100  -batch_size 8  -lr 0.0003 -weight_decay 0.1  -ES_pat 100 -wandb  -time_enc concat "
# python Main.py  $COEFS $SETTING $COMMON $TE__markmc -user_prefix "[$USER_PREFIX]" &

# COMMON="   -epoch 30 -per 100 -batch_size 8  -lr 0.0003 -weight_decay 0.1  -ES_pat 100 -wandb  -time_enc sum "
# python Main.py  $COEFS $SETTING $COMMON $TE__markmc -user_prefix "[$USER_PREFIX]" &

# python Main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &
# python Main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &
# python Main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &
# python Main.py  $COEFS $SETTING $COMMON $TEDA__label -user_prefix "[$USER_PREFIX]" &



for i_split in {0..0}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split -data_label multiclass" 
    SETTING=" -data  $PRE/$DATA_NAME/                 -data_label multiclass" 
    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split -data_label multilabel" 

    # echo $SETTING

    COMMON=" -w_pos -epoch 100 -per 100  -batch_size 256  -lr 0.003 -weight_decay 0.1  -ES_pat 100 -wandb  -time_enc concat "
    python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" &

    # COMMON="  -w_pos -epoch 100 -per 100 -batch_size 256  -lr 0.003 -weight_decay 0.1  -ES_pat 100 -wandb  -time_enc sum "
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" &

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


