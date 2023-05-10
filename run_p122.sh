#!/bin/bash

# python Main.py -data_label multilabel -user_prefix  Sepsis_  -event_enc 1 -state -mod None           -mark_detach 1  -w_pos  -sample_label 1     -int_dec sahp -per 50 -epoch 30


# -demo  -unbalanced_batch
COMMON="-data C:/DATA/data/processed/p12_full_seft/ -epoch 15 -per 50 -w_pos -batch_size 64  -lr 0.00245 "

python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal64]TE__label-'



# # python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[pos2-off-bal16]DA__label-'
# python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal64]TEDA__label-'



# Representation learning
#1     TE+DA->shp+mark
python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1   -unbalanced_batch            -user_prefix '[ubal64]TEDA__shpmark-' 
#2     TE->shp+mark
python Main.py $COMMON -event_enc 1        -mod single -next_mark 1    -unbalanced_batch           -user_prefix '[ubal64]TE__shpmark-'
#3     DA->shp+mark
python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1     -unbalanced_batch          -user_prefix '[ubal64]DA__shpmark-'

# Transfer learning

#     Baseline (no TL, start from scratch)
#         DA-> label
python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal64]DA__label-'
python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal64]TEDA__label-'
python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal64]TE__label-'

python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal64]TEDA__marklabel-'
python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal64]TE__marklabel-'
#     Freezed
#         [#1]-> label
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze ''  -sample_label 1     -user_prefix '[bal64][[TEDA]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'DA'    -sample_label 1     -user_prefix '[bal64][[TE]DA__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'TE'    -sample_label 1     -user_prefix '[bal64][TE[DA]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'TEDA'      -sample_label 1     -user_prefix '[bal64][TEDA__shpmark]__label-'

#         [#2]-> label
python Main.py $COMMON -transfer_learning "TE__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal64][[TE]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TE__shpmark"   -freeze 'TE'      -sample_label 1     -user_prefix '[bal64][TE__shpmark]__label-'

#         [#3]-> label  
python Main.py $COMMON -transfer_learning "DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal64][[DA]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[bal64][DA__shpmark]__label-'




COMMON="-data C:/DATA/data/processed/physio2019_1d_HP_std_AB/ -epoch 15 -per 50 -w_pos -batch_size 8  -lr 0.00245 "

# # python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[pos2-off-bal16]DA__label-'
# python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal64]TEDA__label-'



# Representation learning
#1     TE+DA->shp+mark
python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1   -unbalanced_batch            -user_prefix '[ubal64]TEDA__shpmark-' 
#2     TE->shp+mark
python Main.py $COMMON -event_enc 1        -mod single -next_mark 1    -unbalanced_batch           -user_prefix '[ubal64]TE__shpmark-'
#3     DA->shp+mark
python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1     -unbalanced_batch          -user_prefix '[ubal64]DA__shpmark-'

# Transfer learning

#     Baseline (no TL, start from scratch)
#         DA-> label
python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal64]DA__label-'
python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal64]TEDA__label-'
python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal64]TE__label-'

python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal64]TEDA__marklabel-'
python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal64]TE__marklabel-'
#     Freezed
#         [#1]-> label
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze ''  -sample_label 1     -user_prefix '[bal64][[TEDA]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'DA'    -sample_label 1     -user_prefix '[bal64][[TE]DA__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'TE'    -sample_label 1     -user_prefix '[bal64][TE[DA]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'TEDA'      -sample_label 1     -user_prefix '[bal64][TEDA__shpmark]__label-'

#         [#2]-> label
python Main.py $COMMON -transfer_learning "TE__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal64][[TE]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "TE__shpmark"   -freeze 'TE'      -sample_label 1     -user_prefix '[bal64][TE__shpmark]__label-'

#         [#3]-> label  
python Main.py $COMMON -transfer_learning "DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal64][[DA]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[bal64][DA__shpmark]__label-'

