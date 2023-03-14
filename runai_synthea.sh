#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

N_JOBS=2

USER_PREFIX=H70

DATA_NAME_1="synthea_200"
DATA_NAME_2="synthea_100"
COMMON=" -data_label multilabel  -epoch 50 -log_freq 5 -per 100    -ES_pat 100 -wandb -wandb_project TEEDAM_unsupervised_timeCat "
HPs="-w_pos -pos_alpha 1 -batch_size 64  -lr 0.003 -weight_decay 1 -te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "


PRE="/scratch/hokarami/data_old"
PRE="/mlodata1/hokarami/tedam"

# without label
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_mc="-event_enc 1          -mod single    -next_mark 1     -mark_detach 1      -sample_label 0"
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"

COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"


# python Main.py -w_pos -batch_size 32 -lr 0.003 -weight_decay 0.1 -te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 -w_sample_label 10000 -w_time 1 -w_event 1 -data /scratch/hokarami/data_old/synthea_200/ -split 0 -data_label multilabel -epoch 50 -per 100 -ES_pat 100 -wandb -event_enc 1 -mod none -next_mark 1 -mark_detach 0 -sample_label 0 -user_prefix "[v200-rand-more-R40-TE__nextmark-concat]" -time_enc concat -wandb_project TEEDAM_unsupervised_timeCat


for i_diag in {-2..1}
do

for i_split in {0..4}
do

    SETTING=" -data  $PRE/$DATA_NAME_1/ -split $i_split -diag_offset $i_diag " 

        
    
    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat-d$i_diag]" -time_enc concat &    

    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-sum]" -time_enc sum &
    





    # SETTING=" -data  $PRE/$DATA_NAME_2/ -split $i_split " 


    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat ]" -time_enc concat &    

    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-sum]" -time_enc sum &




    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-concat ]" -time_enc concat &    

    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-sum]" -time_enc sum &



done


done