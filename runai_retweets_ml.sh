#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

N_JOBS=2

USER_PREFIX=R30

DATA_NAME="retweets_ml"
COMMON=" -data_label multilabel  -epoch 50 -per 100    -ES_pat 100 -wandb "
HPs="-w_pos -batch_size 256  -lr 0.003 -weight_decay 0.1-te_d_mark 8 -te_d_time 8 -te_d_inner 16 -te_d_k 8 -te_d_v 8"


PRE="/scratch/hokarami/new"
# PRE="/scratch/hokarami/data_old"

# without label
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_mc="-event_enc 1          -mod single    -next_mark 1     -mark_detach 1      -sample_label 0"
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"

COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"




for i_split in {0..4}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
        
    
    waitforjobs $N_JOBS
    echo python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat ]" -time_enc concat &
    

    waitforjobs $N_JOBS
    echo python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-sum]" -time_enc sum &
    
done

for i_split in {0..4}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
        
    
    waitforjobs $N_JOBS
    echo python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat ]" -time_enc concat &
    

    waitforjobs $N_JOBS
    echo python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-sum]" -time_enc sum &
    
done

for i_split in {0..4}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
        
    
    waitforjobs $N_JOBS
    echo python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-concat ]" -time_enc concat &
    

    waitforjobs $N_JOBS
    echo python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-sum]" -time_enc sum &
    
done


