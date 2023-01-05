#!/bin/bash


# # for seft-based split   -unbalanced_batch 

COMMON="-data C:/DATA/data/processed/p12_full_seft/ -epoch 10 -per 20 -w_pos -w_pos_label 0.5 -batch_size 8  -lr 0.00245 -cuda -wandb"

python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0 -demo   -sample_label 1    -user_prefix '[wd1]DA__label-' &
python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0 -demo   -sample_label 1    -user_prefix '[wd2]DA__label-' &
python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0 -demo   -sample_label 1    -user_prefix '[wd3]DA__label-' &
python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0 -demo   -sample_label 1    -user_prefix '[wd4]DA__label-'


# # Representation learning
# #1     TE+DA->shp+mark
# python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1               -user_prefix '[ubal]TEDA__shpmark-' 
# #2     TE->shp+mark
# python Main.py $COMMON -event_enc 1        -mod single -next_mark 1               -user_prefix '[ubal]TE__shpmark-'
# #3     DA->shp+mark
# # python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1               -user_prefix '[ubal]DA__shpmark-'

# # Transfer learning

# #     Baseline (no TL, start from scratch)
# #         DA-> label
# python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[ubal]DA__label-'
# python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 1    -sample_label 1            -user_prefix '[ubal]TEDA__marklabel-'
# python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[ubal]TE__marklabel-'

# #     Freezed
# #         [#1]-> label
# python Main.py $COMMON -transfer_learning "[ubal]TEDA__shpmark" -freeze ''  -sample_label 1     -user_prefix '[ubal][[TEDA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[ubal]TEDA__shpmark" -freeze 'DA'    -sample_label 1     -user_prefix '[ubal][[TE]DA__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[ubal]TEDA__shpmark" -freeze 'TE'    -sample_label 1     -user_prefix '[ubal][TE[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[ubal]TEDA__shpmark" -freeze 'TEDA'      -sample_label 1     -user_prefix '[ubal][TEDA__shpmark]__label-'

# #         [#2]-> label
# python Main.py $COMMON -transfer_learning "[ubal]TE__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[ubal][[TE]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[ubal]TE__shpmark"   -freeze 'TE'      -sample_label 1     -user_prefix '[ubal][TE__shpmark]__label-'

# #         [#3]-> label  
# # python Main.py $COMMON -transfer_learning "[ubal]DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[ubal][[DA]__shpmark]__label-'
# # python Main.py $COMMON -transfer_learning "[ubal]DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[ubal][DA__shpmark]__label-'

