#!/bin/bash
waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}


N_JOBS=3
USER_PREFIX=RD74-TL3 #  =RD2-wpos03-Complex2-BiasTrue-Simpler2-bs64-PredictorDO02-
DATA_NAME="p12"
PRE="/mlodata1/hokarami/tedam"
COMMON=" -demo -data_label multilabel  -epoch 50 -per 100    -ES_pat 100 -log_freq 1 -wandb -wandb_project TEEDAM_supervised "


TEE_CONFIG_C1="--te_d_mark 8 --te_d_time 4 --te_d_inner 32 --te_d_k 8 --te_d_v 8 --te_n_head 4 --te_n_layers 4 --te_dropout 0.1"
TEE_CONFIG_C2="--te_d_mark 16 --te_d_time 8 --te_d_inner 64 --te_d_k 16 --te_d_v 16 --te_n_head 4 --te_n_layers 4 --te_dropout 0.1"

DAM_CONFIG_C2="--dam_output_activation relu --dam_output_dims  16 --dam_n_phi_layers 3  --dam_phi_width 128  --dam_phi_dropout 0.2  --dam_n_psi_layers 2  --dam_psi_width 64  --dam_psi_latent_width 128 --dam_dot_prod_dim 64  --dam_n_heads 4  --dam_attn_dropout 0.1  --dam_latent_width 64  --dam_n_rho_layers 2  --dam_rho_width 128  --dam_rho_dropout 0.1  --dam_max_timescale 1000  --dam_n_positional_dims 16 
" 

OPT_HPs="-batch_size 128  -lr 0.001 -weight_decay 0.1 -w_pos_label 1 " # simpler2



HPs="$OPT_HPs $TEE_CONFIG_C1 $DAM_CONFIG_C2"





# TEEDAM with label
TEDA__nextmark="-event_enc 1    -state          -mod none      -next_mark 1     -mark_detach 0      -sample_label 1"
TEDA__pp_single_mark="-event_enc 1    -state          -mod single    -next_mark 1     -mark_detach 0      -sample_label 1"
TEDA__pp_ml="-event_enc 1    -state          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"
TEDAnoise__pp_ml="-event_enc 1    -state -noise          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"
TEDAnoise__pp_single_mark="-event_enc 1    -state -noise          -mod single    -next_mark 1     -mark_detach 0      -sample_label 1"

TEDA__none="-event_enc 1    -state          -mod none        -next_mark 1     -mark_detach 1      -sample_label 1"
TE__none="-event_enc 1             -mod none        -next_mark 1     -mark_detach 1      -sample_label 1"

TE__nextmark="-event_enc 1         -mod none      -next_mark 1     -mark_detach 0      -sample_label 1"
TE__pp_ml="-event_enc 1            -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"

# baseline
DA__base="-event_enc 0    -state          -mod none      -next_mark 1     -mark_detach 1      -sample_label 1"
DAnoise__base="-event_enc 0    -state    -noise      -mod none      -next_mark 1     -mark_detach 1      -sample_label 1"



COEFS="-w_sample_label 100  -w_time 1 -w_event 1"



i_diag=0

for i_split in {0..4}
    do
    SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/ -setting raindrop -split $i_split" 
    TL="-transfer_learning DO "


    # # DA__base
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $DA__base -user_prefix "[$USER_PREFIX-DA__base-concat]" -time_enc concat &    


    
    # # TEDA__none NO TL
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat & 

    # # TEDA__pp_ml NO TL
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat &

    # # TEDA__pp_single_mark NO TL
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat]" -time_enc concat &


    # # TEDA__nextmark NO TL
    # waitforjobs $N_JOBS
    # python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat &

   

done




# ********************************************************* TL


i_diag=0
for i_rep in {0..0}
do
    for i_split in {4..4}
        do
        SETTING=" -diag_offset $i_diag -data  $PRE/$DATA_NAME/ -setting raindrop -split $i_split" 
        TL="-transfer_learning DO "
        
        # TEDA__pp_ml
        waitforjobs $N_JOBS
        TL="-transfer_learning DO -freeze TE -tl_tag RD74-ml3"
        python Main.py $TL  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat &

        # # TEDA__pp_single_mark
        # waitforjobs $N_JOBS
        # TL="-transfer_learning DO -freeze TE -tl_tag RD74-single3"
        # python Main.py $TL $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat]" -time_enc concat &


        # TEDA__nextmark
        waitforjobs $N_JOBS
        TL="-transfer_learning DO -freeze TE -tl_tag RD74-nextmark3"
        python Main.py $TL   $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat &


    done
done

