#!/bin/bash

# COMMON="-data C:/DATA/data/processed/physio2019_1d_HP_std/ -epoch 30 -per 50 -w_pos"

# # Representation learning
# #1     TE+DA->shp+mark
# python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1               -user_prefix '[bal]TEDA__shpmark-' 
# #2     TE->shp+mark
# python Main.py $COMMON -event_enc 1        -mod single -next_mark 1               -user_prefix '[bal]TE__shpmark-'
# #3     DA->shp+mark
# python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1               -user_prefix '[bal]DA__shpmark-'

# # Transfer learning

# #     Baseline (no TL, start from scratch)
# #         DA-> label
# python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal]DA__label-'
# python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal]TEDA__marklabel-'
# python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal]TE__marklabel-'

# #     Freezed
# #         [#1]-> label
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze ''  -sample_label 1     -user_prefix '[bal][[TEDA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze 'DA'    -sample_label 1     -user_prefix '[bal][[TE]DA__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze 'TE'    -sample_label 1     -user_prefix '[bal][TE[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze 'TEDA'      -sample_label 1     -user_prefix '[bal][TEDA__shpmark]__label-'

# #         [#2]-> label
# python Main.py $COMMON -transfer_learning "[bal]TE__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal][[TE]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TE__shpmark"   -freeze 'TE'      -sample_label 1     -user_prefix '[bal][TE__shpmark]__label-'

# #         [#3]-> label  
# python Main.py $COMMON -transfer_learning "[bal]DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal][[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[bal][DA__shpmark]__label-'





















# # for hospital-based split

# COMMON="-data C:/DATA/data/processed/physio2019_1d_HP_std_AB/ -epoch 30 -per 50 -w_pos"

# # Representation learning
# #1     TE+DA->shp+mark
# python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1               -user_prefix '[bal]TEDA__shpmark-' 
# #2     TE->shp+mark
# python Main.py $COMMON -event_enc 1        -mod single -next_mark 1               -user_prefix '[bal]TE__shpmark-'
# #3     DA->shp+mark
# python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1               -user_prefix '[bal]DA__shpmark-'

# # Transfer learning

# #     Baseline (no TL, start from scratch)
# #         DA-> label
# python Main.py $COMMON -event_enc 0 -state -mod none -next_mark 0    -sample_label 1            -user_prefix '[bal]DA__label-'
# python Main.py $COMMON -event_enc 1 -state -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal]TEDA__marklabel-'
# python Main.py $COMMON -event_enc 1        -mod none -next_mark 1    -sample_label 1            -user_prefix '[bal]TE__marklabel-'

# #     Freezed
# #         [#1]-> label
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze ''  -sample_label 1     -user_prefix '[bal][[TEDA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze 'DA'    -sample_label 1     -user_prefix '[bal][[TE]DA__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze 'TE'    -sample_label 1     -user_prefix '[bal][TE[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TEDA__shpmark" -freeze 'TEDA'      -sample_label 1     -user_prefix '[bal][TEDA__shpmark]__label-'

# #         [#2]-> label
# python Main.py $COMMON -transfer_learning "[bal]TE__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal][[TE]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]TE__shpmark"   -freeze 'TE'      -sample_label 1     -user_prefix '[bal][TE__shpmark]__label-'

# #         [#3]-> label  
# python Main.py $COMMON -transfer_learning "[bal]DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[bal][[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[bal]DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[bal][DA__shpmark]__label-'

























# # unbalanced case


# COMMON="-data C:/DATA/data/processed/physio2019_1d_HP_std/ -epoch 30 -per 50 -w_pos -unbalanced_batch"

# # Representation learning
# #1     TE+DA->shp+mark
# python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1               -user_prefix '[ubal]TEDA__shpmark-' 
# #2     TE->shp+mark
# python Main.py $COMMON -event_enc 1        -mod single -next_mark 1               -user_prefix '[ubal]TE__shpmark-'
# #3     DA->shp+mark
# python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1               -user_prefix '[ubal]DA__shpmark-'

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
# python Main.py $COMMON -transfer_learning "[ubal]DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[ubal][[DA]__shpmark]__label-'
# python Main.py $COMMON -transfer_learning "[ubal]DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[ubal][DA__shpmark]__label-'





















# for hospital-based split
COMMON="-data C:/DATA/data/processed/physio2019_1d_HP_std_AB/ -epoch 30 -per 50 -w_pos -unbalanced_batch"

# # Representation learning
# #1     TE+DA->shp+mark
# python Main.py $COMMON -event_enc 1 -state -mod single -next_mark 1               -user_prefix '[ubal]TEDA__shpmark-' 
# #2     TE->shp+mark
# python Main.py $COMMON -event_enc 1        -mod single -next_mark 1               -user_prefix '[ubal]TE__shpmark-'
# #3     DA->shp+mark
# python Main.py $COMMON -event_enc 0 -state -mod single -next_mark 1               -user_prefix '[ubal]DA__shpmark-'

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

#         [#2]-> label
python Main.py $COMMON -transfer_learning "[ubal]TE__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[ubal][[TE]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "[ubal]TE__shpmark"   -freeze 'TE'      -sample_label 1     -user_prefix '[ubal][TE__shpmark]__label-'

#         [#3]-> label  
python Main.py $COMMON -transfer_learning "[ubal]DA__shpmark"   -freeze ''    -sample_label 1     -user_prefix '[ubal][[DA]__shpmark]__label-'
python Main.py $COMMON -transfer_learning "[ubal]DA__shpmark"   -freeze 'DA'    -sample_label 1     -user_prefix '[ubal][DA__shpmark]__label-'

