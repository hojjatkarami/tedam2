#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

N_JOBS=4

USER_PREFIX=G70-pos

DATA_NAME="data_so"
COMMON=" -data_label multiclass  -epoch 50 -per 100    -ES_pat 100 -wandb -wandb_project TEEDAM_unsupervised_timeCat "
HPs="-batch_size 4  -lr 0.001 -weight_decay 0.1 -te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "


PRE="/scratch/hokarami/data_old"
PRE="/mlodata1/hokarami/tedam"

# without label
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_mc="-event_enc 1          -mod mc    -next_mark 1     -mark_detach 1      -sample_label 0"
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"

COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"

for i_diag in {0..0}
do

for i_split in {0..4}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split -diag_offset $i_diag " 
        
    
    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat-d$i_diag]" -time_enc concat &
    

    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-sum-d$i_diag]" -time_enc sum &




    
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat-d$i_diag]" -time_enc concat &
    

    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-sum-d$i_diag]" -time_enc sum &
    


    
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_mc -user_prefix "[$USER_PREFIX-TE__pp_mc-concat-d$i_diag]" -time_enc concat &
    

    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_mc -user_prefix "[$USER_PREFIX-TE__pp_mc-sum-d$i_diag]" -time_enc sum &

    
done



done