#!/bin/bash


COMMON="-data C:/DATA/data/processed/physio2019_1d_HP_std_AB/ -epoch 30 -per 50 -w_pos -unbalanced_batch -batch_size 8"


python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix 'DA__label-'


# # Representation learning
# #1     TE+DA->shp+mark
# python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1               -user_prefix 'TEDA__shpmark-' 
# #2     TE->shp+mark
# python Main.py $COMMON -event_enc 1        -mod single -next_mark 1               -user_prefix 'TE__shpmark-'
# #3     DA->shp+mark
# python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1               -user_prefix 'DA__shpmark-'

# # Transfer learning

# #     Baseline (no TL, start from scratch)
# #         DA-> label
# python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix 'DA__label-'
# python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 1    -sample_label 1            -user_prefix 'TEDA__marklabel-'
# python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix 'TE__marklabel-'
# #     Freezed
# #         [#1]-> label
# python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze ''  -sample_label 1     -user_prefix '[[TEDA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'DA'    -sample_label 1     -user_prefix '[[TE]DA__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'TE'    -sample_label 1     -user_prefix '[TE[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "TEDA__shpmark" -freeze 'TEDA'      -sample_label 1     -user_prefix '[TEDA__shpmark]__label-'

# #         [#2]-> label
# python Main.py $COMMON -transfer_learning "TE__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[[TE]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "TE__shpmark"   -freeze 'TE'      -sample_label 1     -user_prefix '[TE__shpmark]__label-'

# #         [#3]-> label  
# python Main.py $COMMON -transfer_learning "DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[DA__shpmark]__label-'


