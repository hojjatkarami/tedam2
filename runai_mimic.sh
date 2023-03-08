#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

N_JOBS=1

USER_PREFIX=H40

DATA_NAME="data_mimic"
COMMON=" -data_label multiclass  -epoch 50 -per 100    -ES_pat 100 -wandb "
HPs="-batch_size 1  -lr 0.0001 -weight_decay 0.1  -te_d_mark 64 -te_d_time 16 -te_d_inner 256 -te_d_k 32 -te_d_v 32 -te_n_head 3 -te_n_layers 3 "


PRE="/scratch/hokarami/data_old"
PRE="/mlodata1/hokarami/tedam"

# without label
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_mc="-event_enc 1          -mod mc    -next_mark 1     -mark_detach 1      -sample_label 0"
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"

COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"




for i_split in {0..4}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
        
    
    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat]" -time_enc concat &    

    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-sum]" -time_enc sum &




    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat]" -time_enc concat &    

    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-sum]" -time_enc sum &




    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_mc -user_prefix "[$USER_PREFIX-TE__pp_mc-concat]" -time_enc concat &
    

    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_mc -user_prefix "[$USER_PREFIX-TE__pp_mc-sum]" -time_enc sum &


    
    
done
