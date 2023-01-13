#!/bin/bash


# # for hospital-based split
#  /codes/data      /scratch/hokarami/data             C:/DATA/data/processed

PRE="/scratch/hokarami/data_tedam"
# PRE="/scratch/hokarami/data"
# PRE="C:/DATA/data/processed"
# PRE="C:/DATA/data/processed"

#p12_full_seft,  p12_full_hosp,                  physio2019_1d_HP_std_AB,          physio2019_1d_HP_std_rand

# declare -i REP=3
DA__label="-event_enc 0 -state       -mod none    -next_mark 0  -sample_label 1 "
TEDA__shpmarklabel="-event_enc 1 -state       -mod single    -next_mark 1  -sample_label 1"
TE__shpmark="-event_enc 1          -mod single    -next_mark 1  -sample_label 0"
TEDA__shpmark="-event_enc 1    -state       -mod single    -next_mark 1  -sample_label 0"



COMMON="    -epoch 50 -per 100 -w_pos -batch_size 8  -lr 0.00245 -weight_decay 0.1  -ES_pat 20 -wandb"


DATA_NAME="p12" 
EXP=" -data  $PRE/$DATA_NAME/ -setting mc2 -test_center 0 " 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-mc2]TEDA__shpmarklabel-" &


DATA_NAME="p19" 
EXP=" -data  $PRE/$DATA_NAME/ -setting mc2 -test_center 0 " 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R3-mc2]TEDA__shpmarklabel-" &


# # # ***********************************************************  sc



# # # ***********************************************************  mc1




# ## TE
# EXP="-setting mc1 -test_center 0 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 1" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 2" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 3" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 4" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" 


# EXP="-setting mc1 -test_center 1 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 1 -split 1" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 1 -split 2" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 1 -split 3" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 1 -split 4" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" 



# EXP="-setting mc1 -test_center 2 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 2 -split 1" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 2 -split 2" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 2 -split 3" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 2 -split 4" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" 


# # ***********************************************************  mc2




# # ## TE
# EXP="-setting mc2 -test_center 0 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc2]TE__shpmark-" &



# EXP="-setting mc2 -test_center 1 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc2]TE__shpmark-" &


# EXP="-setting mc2 -test_center 2 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc2]TE__shpmark-"



# ######################################################################################################################################################

DATA_NAME="p19" 
COMMON="  -data  $PRE/$DATA_NAME/  -epoch 50 -per 100 -w_pos -batch_size 8  -lr 0.00245 -weight_decay 0.1  -ES_pat 20 -wandb"


# # ***********************************************************  sc




# # ***********************************************************  mc1


# EXP="-setting mc1 -test_center 0 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 1" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 2" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 3" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" &

# EXP="-setting mc1 -test_center 0 -split 4" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc1]TE__shpmark-" 


EXP="-setting mc1 -test_center 1 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmark -user_prefix "[CIF-mc1]TEDA__shpmark-" &

EXP="-setting mc1 -test_center 1 -split 1" 
python Main.py $EXP $COMMON $TEDA__shpmark -user_prefix "[CIF-mc1]TEDA__shpmark-" &

EXP="-setting mc1 -test_center 1 -split 2" 
python Main.py $EXP $COMMON $TEDA__shpmark -user_prefix "[CIF-mc1]TEDA__shpmark-" &

EXP="-setting mc1 -test_center 1 -split 3" 
python Main.py $EXP $COMMON $TEDA__shpmark -user_prefix "[CIF-mc1]TEDA__shpmark-" &

EXP="-setting mc1 -test_center 1 -split 4" 
python Main.py $EXP $COMMON $TEDA__shpmark -user_prefix "[CIF-mc1]TEDA__shpmark-" 


EXP="-setting mc2 -test_center 1 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc2]TEDA__shpmarklabel-" &

EXP="-setting mc2 -test_center 1 -split 1" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc2]TEDA__shpmarklabel-" &

EXP="-setting mc2 -test_center 1 -split 2" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc2]TEDA__shpmarklabel-" &

EXP="-setting mc2 -test_center 1 -split 3" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc2]TEDA__shpmarklabel-" &

EXP="-setting mc2 -test_center 1 -split 4" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc2]TEDA__shpmarklabel-" 



# ***********************************************************  mc2



# EXP="-setting mc2 -test_center 0 -split 0" 
# python Main.py $EXP $COMMON $DA__label -user_prefix "[CIF-mc2]DA__label-" &



# EXP="-setting mc2 -test_center 1 -split 0" 
# python Main.py $EXP $COMMON $DA__label -user_prefix "[CIF-mc2]DA__label-" &

# EXP="-setting mc2 -test_center 2 -split 0" 
# python Main.py $EXP $COMMON $DA__label -user_prefix "[CIF-mc2]DA__label-" &

# ## TE
# EXP="-setting mc2 -test_center 0 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc2]TE__shpmark-" &



# EXP="-setting mc2 -test_center 1 -split 0" 
# python Main.py $EXP $COMMON $TE__shpmark -user_prefix "[CIF-mc2]TE__shpmark-" 



# # ## TEDA
# EXP="-setting mc2 -test_center 0 -split 0" 
# python Main.py $EXP $COMMON $TEDA__shpmark -user_prefix "[CIF-mc2]TEDA__shpmark-" &



# EXP="-setting mc2 -test_center 1 -split 0" 
# python Main.py $EXP $COMMON $TEDA__shpmark -user_prefix "[CIF-mc2]TEDA__shpmark-" 



