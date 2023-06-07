#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

N_JOBS=2    

USER_PREFIX=RD-simpler2-SAHPxavier-LRhigher-bs64-

DATA_NAME="p19"
COMMON=" -data_label multilabel  -epoch 100 -per 100    -ES_pat 100 -wandb -wandb_project TEEDAM_unsupervised "
HPs="-batch_size 128  -lr 0.01 -weight_decay 0.1 -te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "  # old
HPs="-batch_size 64  -lr 0.01 -weight_decay 1 -te_d_mark 8 -te_d_time 4 -te_d_inner 32 -te_d_k 8 -te_d_v 8" # simpler2
# HPs="-batch_size 64  -lr 0.01 -weight_decay 1 -te_d_mark 16 -te_d_time 8 -te_d_inner 64 -te_d_k 16 -te_d_v 16" # simpler




PRE="/scratch/hokarami/data_tedam"
PRE="/mlodata1/hokarami/tedam"

# without label
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 2"
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 2"
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 2"
# without label + DAM
TEDA__nextmark="-event_enc 1    -state          -mod none      -next_mark 1     -mark_detach 0      -sample_label 2"
TEDA__pp_single_mark="-event_enc 1    -state          -mod single    -next_mark 1     -mark_detach 0      -sample_label 2"
TEDA__pp_ml="-event_enc 1    -state          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 2"
# without label + noise
TEnoise__nextmark="-event_enc 1      -noise          -mod none      -next_mark 1     -mark_detach 0      -sample_label 2"
TEnoise__pp_single_mark="-event_enc 1      -noise          -mod single    -next_mark 1     -mark_detach 0      -sample_label 2"
TEnoise__pp_ml="-event_enc 1      -noise          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 2"


COEFS="-w_sample_label 100  -w_time 1 -w_event 1"




for i_diag in {0..0}
do






    for i_split in {0..4}
    do
        SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/ -setting raindrop  -split $i_split " 

            # # TE__pp_single_mark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat-d$i_diag]" -time_enc concat &    

            # TEDA__pp_single_mark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat-d$i_diag]" -time_enc concat &    


            # # # TEnoise__pp_single_mark
            # waitforjobs $N_JOBS
            # python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__pp_single_mark -user_prefix "[$USER_PREFIX-TEnoise__pp_single_mark-concat-d$i_diag]" -time_enc concat &    
    done







done # for i_diag
