#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

N_JOBS=4

USER_PREFIX=H10

DATA_NAME="p19"
COMMON=" -data_label multilabel  -epoch 50 -per 100    -ES_pat 100 -wandb -wandb_project TEEDAM_unsupervised "
HPs="-batch_size 128  -lr 0.01 -weight_decay 0.1 -te_d_mark 32 -te_d_time 16 -te_d_inner 128 -te_d_k 32 -te_d_v 32 "



PRE="/scratch/hokarami/data_tedam"
PRE="/mlodata1/hokarami/tedam"

# without label
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"
# without label + DAM
TEDA__nextmark="-event_enc 1    -state          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"
TEDA__pp_single_mark="-event_enc 1    -state          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"
TEDA__pp_ml="-event_enc 1    -state          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"
# without label + noise
TEnoise__nextmark="-event_enc 1      -noise          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"
TEnoise__pp_single_mark="-event_enc 1      -noise          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"
TEnoise__pp_ml="-event_enc 1      -noise          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"


COEFS="-w_sample_label 100  -w_time 1 -w_event 1"



# # random split (rand)
# SETTING=" -data  $PRE/$DATA_NAME/ -setting rand "     

#     # TE__pp_single_mark
#     waitforjobs $N_JOBS
#     python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat]" -time_enc concat &    

#     # TEDA__pp_single_mark
#     waitforjobs $N_JOBS
#     python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat]" -time_enc concat &    


#     # TEnoise__pp_single_mark
#     waitforjobs $N_JOBS
#     python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__pp_single_mark -user_prefix "[$USER_PREFIX-TEnoise__pp_single_mark-concat]" -time_enc concat &    



# multi-center external evaluation split (mc2)    
for i_hosp in {0..2}
do
    SETTING=" -data  $PRE/$DATA_NAME/ -setting mc2 -test_center $i_hosp " 

        # TE__nextmark
        waitforjobs $N_JOBS
        python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat]" -time_enc concat &    

        # TEDA__nextmark
        waitforjobs $N_JOBS
        python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat &    


        # TEnoise__nextmark
        waitforjobs $N_JOBS
        python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__nextmark -user_prefix "[$USER_PREFIX-TEnoise__nextmark-concat]" -time_enc concat &    

done



# multi-center split (mc1)    
for i_hosp in {0..2}
do
    for i_split in {0..4}
    do
        SETTING=" -data  $PRE/$DATA_NAME/ -setting mc1 -test_center $i_hosp -split $i_split " 

            # TE__nextmark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat]" -time_enc concat &    

            # TEDA__nextmark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat &    


            # TEnoise__nextmark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__nextmark -user_prefix "[$USER_PREFIX-TEnoise__nextmark-concat]" -time_enc concat &    
    done
done



# single-center split (sc)    
for i_hosp in {0..2}
do
    for i_split in {0..4}
    do
        SETTING=" -data  $PRE/$DATA_NAME/ -setting sc -test_center $i_hosp -split $i_split " 

            # TE__nextmark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat]" -time_enc concat &    

            # TEDA__nextmark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat &    


            # TEnoise__nextmark
            waitforjobs $N_JOBS
            python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__nextmark -user_prefix "[$USER_PREFIX-TEnoise__nextmark-concat]" -time_enc concat &    
    done
done



# # # seft split (seft)
# # SETTING=" -data  $PRE/$DATA_NAME/ -setting seft "     

# #     # TE__pp_ml
# #     waitforjobs $N_JOBS
# #     python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-concat]" -time_enc concat &    

# #     # TEDA__pp_ml
# #     waitforjobs $N_JOBS
# #     python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat &    


# #     # TEnoise__pp_ml
# #     waitforjobs $N_JOBS
# #     python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__pp_ml -user_prefix "[$USER_PREFIX-TEnoise__pp_ml-concat]" -time_enc concat &    



# # multi-center external evaluation split (mc2)    
# for i_hosp in {0..2}
# do
#     SETTING=" -data  $PRE/$DATA_NAME/ -setting mc2 -test_center $i_hosp " 

#         # TE__pp_ml
#         waitforjobs $N_JOBS
#         python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-concat]" -time_enc concat &    

#         # TEDA__pp_ml
#         waitforjobs $N_JOBS
#         python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat &    


#         # TEnoise__pp_ml
#         waitforjobs $N_JOBS
#         python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__pp_ml -user_prefix "[$USER_PREFIX-TEnoise__pp_ml-concat]" -time_enc concat &    

# done



# # multi-center split (mc1)    
# for i_hosp in {0..2}
# do
#     for i_split in {0..4}
#     do
#         SETTING=" -data  $PRE/$DATA_NAME/ -setting mc1 -test_center $i_hosp -split $i_split " 

#             # TE__pp_ml
#             waitforjobs $N_JOBS
#             python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-concat]" -time_enc concat &    

#             # TEDA__pp_ml
#             waitforjobs $N_JOBS
#             python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat &    


#             # TEnoise__pp_ml
#             waitforjobs $N_JOBS
#             python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__pp_ml -user_prefix "[$USER_PREFIX-TEnoise__pp_ml-concat]" -time_enc concat &    
#     done
# done



# # single-center split (sc)    
# for i_hosp in {0..2}
# do
#     for i_split in {0..4}
#     do
#         SETTING=" -data  $PRE/$DATA_NAME/ -setting sc -test_center $i_hosp -split $i_split " 

#             # TE__pp_ml
#             waitforjobs $N_JOBS
#             python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_ml -user_prefix "[$USER_PREFIX-TE__pp_ml-concat]" -time_enc concat &    

#             # TEDA__pp_ml
#             waitforjobs $N_JOBS
#             python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat &    


#             # TEnoise__pp_ml
#             waitforjobs $N_JOBS
#             python Main.py  $HPs $COEFS $SETTING $COMMON $TEnoise__pp_ml -user_prefix "[$USER_PREFIX-TEnoise__pp_ml-concat]" -time_enc concat &    
#     done
# done
