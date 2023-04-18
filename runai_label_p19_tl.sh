#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

N_JOBS=4

USER_PREFIX=H70HHGGG-

DATA_NAME="p19"
COMMON=" -data_label multilabel  -epoch 50 -per 100    -ES_pat 100 -wandb -wandb_project TEEDAM_supervised "
HPs="-batch_size 128  -lr 0.01 -weight_decay 0.1 -te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "



PRE="/scratch/hokarami/data_tedam"
PRE="/mlodata1/hokarami/tedam"

# TEEDAM with label
TEDA__nextmark="-event_enc 1    -state          -mod none      -next_mark 1     -mark_detach 0      -sample_label 1"
TEDA__pp_single_mark="-event_enc 1    -state          -mod single    -next_mark 1     -mark_detach 0      -sample_label 1"
TEDA__pp_ml="-event_enc 1    -state          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"
TEDAnoise__pp_ml="-event_enc 1    -state -noise          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"
TEDAnoise__pp_single_mark="-event_enc 1    -state -noise          -mod single    -next_mark 1     -mark_detach 0      -sample_label 1"

TEDA__none="-event_enc 1    -state          -mod none        -next_mark 1     -mark_detach 1      -sample_label 1"
TE__none="-event_enc 1    -state          -mod none        -next_mark 1     -mark_detach 1      -sample_label 1"

TE__nextmark="-event_enc 1         -mod none      -next_mark 1     -mark_detach 0      -sample_label 1"
TE__pp_ml="-event_enc 1            -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"

# baseline
DA__base="-event_enc 0    -state          -mod none      -next_mark 1     -mark_detach 1      -sample_label 1"
DAnoise__base="-event_enc 0    -state    -noise      -mod none      -next_mark 1     -mark_detach 1      -sample_label 1"



COEFS="-w_sample_label 100  -w_time 1 -w_event 1"


i_diag=0


# multi-center external evaluation split (mc2)    
for i_hosp in {0..1}
do
    SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/ -setting mc2 -test_center $i_hosp " 
    TL="-transfer_learning DO "
    
    # DA__base
    waitforjobs $N_JOBS
    python Main.py  $HPs $COEFS $SETTING $COMMON $DA__base -user_prefix "[$USER_PREFIX-DA__base-concat]" -time_enc concat &    

        

    # # TE__none
    # TL="-transfer_learning DO -freeze TE"
    # waitforjobs $N_JOBS
    # python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TE__none -user_prefix "[$USER_PREFIX-TE__none-concat]" -time_enc concat & 


    # TEDA__none TL
    TL="-transfer_learning DO -freeze TE"
    waitforjobs $N_JOBS
    python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat & 

    # # TEDA__none NO TL
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat & 

done



# multi-center split (mc1)    
for i_hosp in {0..1}
do
    for i_split in {0..4}
    do
        SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/ -setting mc1 -test_center $i_hosp -split $i_split " 
        TL='-transfer_learning DO '

        # DA__base
        waitforjobs $N_JOBS
        python Main.py  $HPs $COEFS $SETTING $COMMON $DA__base -user_prefix "[$USER_PREFIX-DA__base-concat]" -time_enc concat &    

            

        # # TE__none
        # TL="-transfer_learning DO -freeze TE"
        # waitforjobs $N_JOBS
        # python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TE__none -user_prefix "[$USER_PREFIX-TE__none-concat]" -time_enc concat & 


        # TEDA__none TL
        TL="-transfer_learning DO -freeze TE"
        waitforjobs $N_JOBS
        python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat & 

        # # TEDA__none NO TL
        # waitforjobs $N_JOBS
        # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat & 

    done
done


# single-center split (sc)    
for i_hosp in {0..1}
do
    for i_split in {0..4}
    do
        SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/ -setting sc -test_center $i_hosp -split $i_split " 
        TL='-transfer_learning DO '

        # DA__base
        waitforjobs $N_JOBS
        python Main.py  $HPs $COEFS $SETTING $COMMON $DA__base -user_prefix "[$USER_PREFIX-DA__base-concat]" -time_enc concat &    

            

        # # TE__none
        # TL="-transfer_learning DO -freeze TE"
        # waitforjobs $N_JOBS
        # python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TE__none -user_prefix "[$USER_PREFIX-TE__none-concat]" -time_enc concat & 


        # TEDA__none TL
        TL="-transfer_learning DO -freeze TE"
        waitforjobs $N_JOBS
        python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat & 

        # # TEDA__none NO TL
        # waitforjobs $N_JOBS
        # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat & 


    done
done