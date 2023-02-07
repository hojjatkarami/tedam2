#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}


N_JOBS=2
USER_PREFIX=R5-OLD-100var-NoEvent-sum
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

COEFS="-w_sample_label 10000  -w_time 1 -w_event 0"



    

DATA_NAME="synthea_full"
COMMON="-w_pos -pos_alpha 1 -data_label multilabel  -epoch 20 -per 100  -batch_size 64  -lr 0.003   -ES_pat 100 -wandb "

HPs="-te_d_mark 128 -te_d_time 64 -te_d_inner 512 -te_d_k 64 -te_d_v 64 "
HPs="-te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "
HPs="-te_d_mark 64 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "
HPs="-te_d_mark 8 -te_d_time 8 -te_d_inner 16 -te_d_k 8 -te_d_v 8 "

# HPs="-te_d_mark 8 -te_d_time 8 -te_d_inner 16 -te_d_k 8 -te_d_v 8 "


# [ 8.23580348,2.31585837,36.0073046,11.9821909,39.84079,14.32224406,50.,50.,16.04390244,17.78494624,32.34188878,50.,2.77125205,18.79410041,50.,23.41590361,50.,50.,50.,10.93193594,19.74651925,12.84613282,13.06524153,11.34328176,48.69396763,50.,50.,33.51158038,50.,11.59490367,45.14116576,16.92428799,16.92428799,50.,33.48808713,23.05650522,50.,50.,50.,22.30940879,21.25477707,21.25477707,50.,50.,50.,50.,13.12996793,50.,50.,50. ]
 
for i_split in {0..4}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
    
    

    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat -w_pos -pos_alpha 0.02 &

    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat -w_pos -pos_alpha 0.1 &

    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat  -weight_decay 0.1 &
    
    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat  -weight_decay 0.01 &

    waitforjobs $N_JOBS
    python Main.py $HPs $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat  -weight_decay 0.1 &



    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat  &
    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat  -pos_alpha 0.5 &


    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc concat  -pos_alpha 0.2 &



    # waitforjobs $N_JOBS
    # python Main.py  $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX]" -time_enc sum &


done


