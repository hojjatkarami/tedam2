# %%
# activate line execution
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# general
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import shutil
import pickle

# plotly
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import custom libraries
import sys
sys.path.append("C:\\DATA\\Tasks\\lib\\hk")
import hk_utils

# folder paths
ADD_DATA = "C:\\DATA\\data\\raw\\mimic4\\lookup\\"
ADD_DATA_proc = "C:/DATA/data/processed/"


IMG_PATH_PAPER = "C:\\DATA\\Tasks\\220704\\Alternate-Transactions-Articles-LaTeX-template\\images\\"


# %%
# libraries for THP

import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

import transformer.Constants as Constants
import Utils

# from preprocess.Dataset import get_dataloader, get_dataloader2
# from transformer.Models import Transformer
# from transformer.hk_transformer import Transformer
from tqdm import tqdm

# from torchinfo import summary

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# torch.cuda.empty_cache()
# torch.cuda.memory_allocated()
# torch.cuda.memory_reserved()

from sklearn import metrics
# from hk_pytorch import save_checkpoint,load_checkpoint
# import hk_pytorch


# from custom2 import myparser
import re

# %%
import Main
import webbrowser


# %%
from tsnecuda import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

# %%
TSNE

# %% [markdown]
# # FUnctions

# %%
def find_folder(path_dir, sub_string):

    os.chdir(path_dir)



    candidateFolders = filter(lambda x: (sub_string in x), os.listdir(path_dir))
    candidateFolders = [f for f in candidateFolders]
    candidateFolders.sort(key=lambda x: os.path.getmtime(x)) # newest last

    print(candidateFolders)


    return path_dir+candidateFolders[-1]

# %% [markdown]
# # Model path

# %%
add_data = "C:/DATA/data/processed/physio2019_1d_HP_std/"

run_names = [
            # '[bal]DA__label-877287-none-state1-per50',
            
            # '[bal]TEDA__marklabel-877399-none-state1-per50',
            
            # '[bal]TE__marklabel-877594-none-state0-per50',
            
            # '[bal][DA__shpmark]__label-878561-single-state1-per50',
            # '[bal][TEDA__shpmark]__label-878158-single-state1-per50',
            # '[bal][TE[DA]__shpmark]__label-877987-single-state1-per50',
            # '[bal][TE__shpmark]__label-878341-single-state0-per50',
            # '[bal][[DA]__shpmark]__label-878474-single-state1-per50',
            # '[bal][[TEDA]__shpmark]__label-877720-single-state1-per50',
            # '[bal][[TE]DA__shpmark]__label-877831-single-state1-per50',
            # '[bal][[TE]__shpmark]__label-878266-single-state0-per50',
            '[ubal]DA__label-882519-none-state1-per50',
          
            '[ubal]TEDA__marklabel-882603-none-state1-per50',
            
            '[ubal]TE__marklabel-882768-none-state0-per50',
           
            '[ubal][TEDA__shpmark]__label-883278-single-state1-per50',
            '[ubal][TE[DA]__shpmark]__label-883114-single-state1-per50',
            '[ubal][TE__shpmark]__label-883447-single-state0-per50',
            '[ubal][[DA]__shpmark]__label-883591-single-state1-per50',
            '[ubal][[TEDA]__shpmark]__label-882889-single-state1-per50',
            '[ubal][[TE]DA__shpmark]__label-882985-single-state1-per50',
            '[ubal][[TE]__shpmark]__label-883375-single-state0-per50'
]



for run_name in run_names:
    # run_name = run_name[-1]
    run_address = find_folder(path_dir=add_data, sub_string=run_name)
    run_address


    # find last epoch
    all_epochs=[]
    for file in os.listdir(run_address):
        if 'model_ep' in file:
            all_epochs.append(int(re.findall(r'\d+', file)[0]))
    MODEL_PATH = run_address+'/model_ep'+str(max(all_epochs))+'.pkl'
    MODEL_PATH


    # load opt file
    opt = pickle.load(    open(run_address+'/opt.pkl','rb')  )

    # modify opt
    opt = Main.config(opt, justLoad=True)



    # train_loader, test_loader, additional_info = Main.prepare_dataloader(opt)



    # %% [markdown]
    # ## load pre-trained model

    # %%
    model = Main.ATHP(
            n_marks=opt.num_marks,
            TE_config = opt.TE_config,
            DAM_config = opt.DAM_config,

            CIF_config = opt.CIF_config,
            next_time_config = opt.next_time_config,
            next_type_config = opt.next_type_config,
            label_config = opt.label_config,



        )
    model.to(opt.device)



    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                            opt.lr, betas=(0.9, 0.999), eps=1e-05)

    # update model and optimizer with the last checkpoint    
    checkpoint = torch.load(MODEL_PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    _ = model.eval()

    # %%
    DAM_log=[]
    enc_log=[]
    state_label_log = []

    total_label_state=0
    y_state_pred_list = []
    y_state_true_list = []
    y_state_score_list = []
    a=[]
    b=[]

    att_log=[]

    with torch.no_grad():
        for batch in tqdm(opt.testloader, mininterval=2,
                        desc='  - (Testing)   ', leave=False):


            if opt.state or opt.sample_label:
                # event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch[0])
                # new = next(iter_stateloader)
                event_time, time_gap, event_type, state_time, state_value, state_mod, state_label = map(lambda x: x.to(opt.device), batch)
                enc_out = model(event_type, event_time, state_time, state_value, state_mod, verbose=True)
            else:
                event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
                enc_out = model(event_type, event_time, verbose=True)

            non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(2)
            if opt.sample_label:
                state_label_red = Main.align(state_label[:,:,None].float(), event_time, state_time) # [B,L,1]
                # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]

                if (state_label_red.sum(1).bool().squeeze(-1).float() - state_label.sum(1).bool().int()).sum() !=0:
                    aaa=1

                state_label_loss, (y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)
                total_label_state += state_label_loss.item()
                # y_state_pred_list.append( torch.flatten(y_state_pred, end_dim=1) ) # [*]
                # y_state_true_list.append( torch.flatten(y_state_true, end_dim=1) ) # [*]
                # y_state_score_list.append( torch.flatten(y_state_score, end_dim=1) ) # [*, C]


                y_state_pred_list.append(y_state_pred) # [*]
                y_state_true_list.append(y_state_true) # [*]
                y_state_score_list.append(y_state_score) # [*] it is bina

            # a.append(y_state_pred.cpu().numpy())
            # b.append(model.temp['samp_pred'])
            # if (model.temp['samp_pred']-y_state_pred.cpu().numpy()   ).sum() !=0:
            #     print('FUCK')
            #     aa=1
            # DAM_log.append(model.DAM.temp['att'])
            enc_log.append(model.temp['enc_last'])
            # model.temp
            state_label_log.append(state_label.sum(-1).bool().int().cpu().numpy())

            


    # %%
    y_state_pred = (torch.cat(y_state_pred_list)).cpu().numpy() # [*]
    y_state_true = (torch.cat(y_state_true_list)).cpu().numpy()
    y_state_score = (torch.cat(y_state_score_list)).cpu().numpy()

    dict_metrics={}
    dict_metrics.update({
                # 'sample_label/loss':total_label_state/total_num_event,
                'sample_label/f1-binary': metrics.f1_score(y_state_true, y_state_pred ,average='binary', zero_division=0),
                'sample_label/recall-binary': metrics.recall_score(y_state_true, y_state_pred ,average='binary', zero_division=0),
                'sample_label/precision-binary': metrics.precision_score(y_state_true, y_state_pred ,average='binary', zero_division=0),

                'sample_label/ACC': metrics.accuracy_score(y_state_true, y_state_pred),

                'sample_label/AUROC': metrics.roc_auc_score(y_state_true, y_state_score),
                'sample_label/AUPRC': metrics.average_precision_score(y_state_true, y_state_score),

                # 'sample_label/sum_r_enc':torch.concat(r_enc_list, dim=1).sum().item(),
                # 'sample_label/r_enc_zero': a,
            })
    conf_mat = metrics.confusion_matrix(y_state_true, y_state_pred)
    conf_mat
    dict_metrics
    print(metrics.classification_report(y_state_true, y_state_pred, zero_division=0))

    # %% [markdown]
    # # Visualizations

    # %% [markdown]
    # ## TSNE of Learned Rep

    # %%
    TSNE_LIMIT = 2000

    x = np.concatenate(enc_log,axis=0)

    x.shape
    all_colors = ["#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9"]

    tsne = TSNE(n_components=2, perplexity=15, learning_rate=10,n_jobs=4)
    X_tsne = tsne.fit_transform(x[:TSNE_LIMIT,:])
    # colors_tsne = [all_colors[label] for label in dict_metrics['tsne']['y_true'][:TSNE_LIMIT] ]
    X_tsne.shape


    # %%
    # x[:TSNE_LIMIT,:].mean()
    # x[:TSNE_LIMIT,:].std()
    # # px.density_heatmap(x[:TSNE_LIMIT,:])

    # px.violin(x[:TSNE_LIMIT,:], box=False)

    # %%
    df = pd.DataFrame()
    df['x']=X_tsne[:,0]
    df['y']=X_tsne[:,1]

    TP = (y_state_true[:TSNE_LIMIT]*y_state_pred[:TSNE_LIMIT])==1
    FP_FN = (y_state_true[:TSNE_LIMIT]+y_state_pred[:TSNE_LIMIT])==1
    TN = (y_state_true[:TSNE_LIMIT]+y_state_pred[:TSNE_LIMIT])==0


    df['color']=0
    df['id']=np.arange(len(df))
    # df.loc[TN, 'color']='True Negatives'
    # df.loc[TP, 'color']='True Positives'
    # df.loc[FP_FN, 'color']='False Classified'

    df.loc[y_state_true[:TSNE_LIMIT].astype(bool).flatten(), 'color']='Positive Samples'
    df.loc[~y_state_true[:TSNE_LIMIT].astype(bool).flatten(), 'color']='Negative Samples'

    # df.loc[y_state_pred[:TSNE_LIMIT].astype(bool), 'color']='Negative Samples'

    # df['color_true']=y_state_true[:TSNE_LIMIT]
    # df['color_pred']=y_state_pred[:TSNE_LIMIT]

    conf_mat
    fig = px.scatter(df,x='x',y='y',color='color', title=run_name, hover_data=['id']).update_layout(width=800, height=600)
    fig.show()
    fig.write_image(IMG_PATH_PAPER+f"tsne_{run_name}.png")
    # px.scatter(df,x='x',y='y',color='color_pred').update_layout(width=600, height=600).show()
