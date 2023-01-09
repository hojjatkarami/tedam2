#!/bin/bash


# # for hospital-based split
#  /codes/data      /scratch/hokarami/data             C:/DATA/data/processed

PRE="/scratch/hokarami/data_tedam"
# PRE="/scratch/hokarami/data"
# PRE="C:/DATA/data/processed"
# PRE="C:/DATA/data/processed"

#p12_full_seft,  p12_full_hosp,                  physio2019_1d_HP_std_AB,          physio2019_1d_HP_std_rand

# declare -i REP=3
DA__label="-event_enc 0 -state       -mod none    -next_mark 0  -sample_label 1                 -lr 0.002701      -weight_decay 0.08159                     "
TEDA__shpmarklabel="-event_enc 1 -state       -mod single    -next_mark 1  -sample_label 1       -lr 0.001346      -weight_decay 0.01385                      "



DATA_NAME="p19" 
COMMON="  -data  $PRE/$DATA_NAME/  -epoch 50 -per 100 -w_pos -batch_size 8    -ES_pat 20 -wandb"

# # ***********************************************************  sc

EXP="-setting sc -test_center 0 -split 0" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 0 -split 1" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 0 -split 2" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 0 -split 3" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 0 -split 4" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" 


EXP="-setting sc -test_center 1 -split 0" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 1 -split 1" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 1 -split 2" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 1 -split 3" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" &

EXP="-setting sc -test_center 1 -split 4" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-sc]DA__label-" 



## TEDAM
EXP="-setting sc -test_center 0 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 0 -split 1" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 0 -split 2" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 0 -split 3" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 0 -split 4" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" 


EXP="-setting sc -test_center 1 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 1 -split 1" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 1 -split 2" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 1 -split 3" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" &

EXP="-setting sc -test_center 1 -split 4" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-sc]TEDA__shpmarklabel-" 










# ***********************************************************  mc1



EXP="-setting mc1 -test_center 0 -split 0" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 0 -split 1" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 0 -split 2" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 0 -split 3" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 0 -split 4" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" 


EXP="-setting mc1 -test_center 1 -split 0" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 1 -split 1" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 1 -split 2" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 1 -split 3" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" &

EXP="-setting mc1 -test_center 1 -split 4" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc1]DA__label-" 



## TEDAM
EXP="-setting mc1 -test_center 0 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 0 -split 1" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 0 -split 2" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 0 -split 3" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 0 -split 4" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" 


EXP="-setting mc1 -test_center 1 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 1 -split 1" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 1 -split 2" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 1 -split 3" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" &

EXP="-setting mc1 -test_center 1 -split 4" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc1]TEDA__shpmarklabel-" 





# ***********************************************************  mc2



EXP="-setting mc2 -test_center 0 -split 0" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc2]DA__label-" &



EXP="-setting mc2 -test_center 1 -split 0" 
python Main.py $EXP $COMMON $DA__label -user_prefix "[R2-mc2]DA__label-" &




## TEDAM
EXP="-setting mc2 -test_center 0 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc2]TEDA__shpmarklabel-" &



EXP="-setting mc2 -test_center 1 -split 0" 
python Main.py $EXP $COMMON $TEDA__shpmarklabel -user_prefix "[R2-mc2]TEDA__shpmarklabel-" &



