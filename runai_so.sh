#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}


N_JOBS=2
USER_PREFIX=R4
# p12     -lr 0.001 -weight_decay 0.001  
# p19     -lr 0.001 -weight_decay 1    #DA__label   TE__shpmark

# # for hospital-based split
#  /codes/data      /scratch/hokarami/data             C:/DATA/data/processed

PRE="/scratch/hokarami/data_tedam"
PRE="C:/DATA/data/processed"
PRE="/scratch/hokarami/new"
# PRE="/scratch/hokarami/data_old"




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

COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"



DATA_NAME="data_so"
COMMON=" -data_label multiclass  -epoch 30 -per 100  -batch_size 8  -lr 0.0003 -weight_decay 0.1  -ES_pat 100 -wandb "
HPs="-te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "
 
for i_split in {0..0}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
        
    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat &
    
    waitforjobs $N_JOBS
    echo python Main.py  $HPs $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX-wclass]" -time_enc concat -w_class&
    
    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc sum &


done
    
