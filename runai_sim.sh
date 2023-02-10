#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

DATA_NAME="sahp_sim"
N_JOBS=1
USER_PREFIX=R10-Nsplit

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



    


COMMON="-data_label multiclass  -epoch 10 -per 100  -batch_size 16  -lr 0.003 -weight_decay 0.1  -ES_pat 100 -wandb "

HPs="-te_d_mark 128 -te_d_time 64 -te_d_inner 512 -te_d_k 64 -te_d_v 64 "
HPs="-te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "
HPs="-te_d_mark 64 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "
HPs="-te_d_mark 8 -te_d_time 8 -te_d_inner 16 -te_d_k 8 -te_d_v 8 "
HPs="-te_d_mark 8 -te_d_time 8 -te_d_inner 16 -te_d_k 8 -te_d_v 8 "
HPs="-te_d_mark 4 -te_d_time 4 -te_d_inner 16 -te_d_k 4 -te_d_v 4 -te_n_head 1 -te_n_layers 1"


 
# for i_split in {0..4}
# do

#     SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
    
    

#     waitforjobs $N_JOBS
#     python Main.py $HPs $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX concat]" -time_enc concat  &

#     waitforjobs $N_JOBS
#     python Main.py $HPs $COEFS $SETTING $COMMON $TE__shpmark -user_prefix "[$USER_PREFIX sum]" -time_enc sum  &


# done

for i_split in {0..0}
do

    SETTING=" -data  $PRE/$DATA_NAME/ -split $i_split " 
    
    

    # waitforjobs $N_JOBS
    # python Main.py $HPs $COEFS $SETTING $COMMON $TE__markmc -user_prefix "[$USER_PREFIX TE__markmc concat]" -time_enc concat  &

    waitforjobs $N_JOBS
    python Main.py $HPs $COEFS $SETTING $COMMON $TE__markmc -user_prefix "[$USER_PREFIX TE__markmc sum]" -time_enc sum  


done

