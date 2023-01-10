import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader, get_dataloader_state, get_dataloader2, get_dataloader3
# from preprocess.Dataset_mhp import get_dataloader
# from preprocess.Dataset_mhp import prepare_dataloader

from transformer.Models import Transformer, ATHP, align
from tqdm import tqdm

import datetime
import os, glob, re
import shutil

import wandb
os.environ["WANDB_API_KEY"] = "0f780ac8a470afe6cb7fc474ff3794772c660465"
os.environ["WANDB_START_METHOD"] = "thread"

import optuna

from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics


import matplotlib.pyplot as plt


from collections import OrderedDict


from torch.profiler import profile, record_function, ProfilerActivity


# import custom libraries
import sys
# sys.path.append("C:\\DATA\\Tasks\\lib\\hk")
# # import hk_psql
# import hk_utils

def write_to_summary(dict_metrics, opt, i_epoch=-1, prefix=''):
    TSNE_LIMIT = 1000
    all_colors = ["#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
    "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9"]

    if 'ConfMat' in dict_metrics:
        # fig = plt.figure(figsize=(8,8))
        fig, ax = plt.subplots(figsize=(10,10))
        dict_metrics['ConfMat'].plot(ax=ax)
        # dict_metrics['ConfMat'].plot()
        opt.writer.add_figure('matplotlib', fig, i_epoch)
        dict_metrics.pop('ConfMat')
        # plt.close()

    if 'time_gap_data' in dict_metrics:
        fig, ax = plt.subplots(figsize=(10,10))
        _ = ax.scatter(dict_metrics['time_gap_data'][0][1:], dict_metrics['time_gap_data'][1][:-1])
        _ = ax.plot([0 ,2], [0, 2],'r-')
        opt.writer.add_figure('time_gap', fig, i_epoch)
        dict_metrics.pop('time_gap_data')
        # plt.close()

    if 'tsne' in dict_metrics:


        if (opt.i_epoch) % opt.write2tsne == 0:
            with open(f"{opt.add_model}tsne_epoch_{i_epoch}.pkl", 'wb') as handle:
                pickle.dump(dict_metrics['tsne'], handle)

            tsne = TSNE(n_components=2, perplexity=15, learning_rate=10,n_jobs=4)

            if (opt.i_epoch ) % opt.write2tsne == 0 :
                # with hk_utils.Timer('TSNE COMPUTATIOIN:'):
                # X_tsne = tsne.fit_transform(X_enc)
                print('TSNE')
                X_tsne = tsne.fit_transform(dict_metrics['tsne']['X_enc'][:TSNE_LIMIT,:])

            colors_tsne = [all_colors[label] for label in dict_metrics['tsne']['y_true'][:TSNE_LIMIT] ]
            fig, ax = plt.subplots(figsize=(10,10))
            _ = ax.scatter(X_tsne[:TSNE_LIMIT,0], X_tsne[:TSNE_LIMIT,1], c=colors_tsne)

            # _ = ax.scatter(dict_metrics['tsne']['X_tsne'][:,0], dict_metrics['tsne']['X_tsne'][:,1], c=colors_tsne)
            opt.writer.add_figure('tsne', fig, i_epoch)
        dict_metrics.pop('tsne')

    for k,v in dict_metrics.items():
        # opt.writer.add_scalars('Metrics/'+k, {'test':v}, i_epoch)
        # if '/' in k:
        #     prefix = 'Test-'
        # else:
        #     prefix = 'Test/'
        if isinstance(v,np.ndarray):
            opt.writer.add_histogram(prefix+k, v, i_epoch)
        else:
            opt.writer.add_scalar(prefix+k, v, i_epoch)

        if opt.wandb:
            # dict_metrics.update({'i_epoch':i_epoch})
            wandb.log({(prefix+k):v}, step=i_epoch)
    # for k,v in dict_roc_auc.items():
        # opt.writer.add_scalars('Metrics/'+k, {'test':v}, i_epoch)
    # opt.writer.add_scalars('Metrics/ROC', dict_roc_auc, i_epoch)



def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        additional_info = {}

        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            # num_types = int(data['dim_process'])


            if 'dim_process' in data:
                additional_info['num_types'] = data['dim_process']
            if 'num_marks' in data:
                additional_info['num_marks'] = data['num_marks']
            if 'dict_map_events' in data:
                additional_info['dict_map_events'] = data['dict_map_events']
            if 'pos_weight' in data:
                additional_info['pos_weight'] = data['pos_weight']
            if 'w_class' in data:
                additional_info['w'] = data['w_class']
            if 'dict_map_states' in data:
                additional_info['dict_map_states'] = data['dict_map_states']
            if 'num_states' in data:
                additional_info['num_states'] = data['num_states']

        return data[dict_name], additional_info

    def load_data2(name):
        additional_info = {}

        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            # num_types = int(data['dim_process'])


            if 'dim_process' in data:
                additional_info['num_types'] = data['dim_process']
            if 'num_marks' in data:
                additional_info['num_marks'] = data['num_marks']
            if 'dict_map_events' in data:
                additional_info['dict_map_events'] = data['dict_map_events']
            if 'pos_weight' in data:
                additional_info['pos_weight'] = data['pos_weight']
            if 'w_class' in data:
                additional_info['w'] = data['w_class']
            if 'dict_map_states' in data:
                additional_info['dict_map_states'] = data['dict_map_states']
            if 'num_states' in data:
                additional_info['num_states'] = data['num_states']
            if 'num_demos' in data:
                additional_info['num_demos'] = data['num_demos']

        return data, additional_info
    print('[Info] Loading train data...')
    train_data, additional_info = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    valid_data, _= load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')


    if opt.per>0:
        print(f'[info] {opt.per}% of data will be considered')
        train_data = train_data[:int(opt.per/100*len(train_data))]
        # test_data = test_data[:int(opt.per/100*len(test_data))]


    no_modes = 1

    train_state=None
    test_state=None
    valid_state=None



    if opt.state or opt.sample_label:

        # print('[Info] Loading train STATE...')
        # train_state, new_additional_info = load_data(opt.data + 'train_state.pkl', 'train')
        # print('[Info] Loading dev STATE...')
        # dev_state,_= load_data(opt.data + 'dev_state.pkl', 'dev')
        # print('[Info] Loading test STATE...')
        # test_state,_ = load_data(opt.data + 'test_state.pkl', 'test')

        print('[Info] Loading train STATE...')
        train_state, new_additional_info = load_data2(opt.data + 'train_state.pkl')
        print('[Info] Loading dev STATE...')
        valid_state,_= load_data2(opt.data + 'dev_state.pkl')
        print('[Info] Loading test STATE...')
        test_state,_ = load_data2(opt.data + 'test_state.pkl')


        additional_info.update(new_additional_info)
        if opt.per>0:
            print(f'[info] {opt.per}% of data will be considered')
            train_state['state'] = train_state['state'][:int(opt.per/100*len(train_state['state']))]
            # test_state['state'] = test_state['state'][:int(opt.per/100*len(test_state['state']))]
            if 'demo' in train_state.keys():
                train_state['demo'] = train_state['demo'][:int(opt.per/100*len(train_state['demo']))]
                # test_state['demo'] = test_state['demo'][:int(opt.per/100*len(test_state['demo']))]

    #     trainloader = get_dataloader2(train_data, train_state, opt.batch_size, shuffle=True, data_label=opt.data_label, balanced=opt.balanced_batch)
    #     testloader = get_dataloader2(test_data, test_state, opt.batch_size, shuffle=False, data_label=opt.data_label, balanced=False)
    # else:
    #     trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True, data_label=opt.data_label, balanced=opt.balanced_batch)
    #     testloader = get_dataloader(test_data, opt.batch_size, shuffle=False, data_label=opt.data_label, balanced=False)

    state_args = {'have_label':opt.sample_label, 'have_demo':opt.demo}
    
    trainloader = get_dataloader3(train_data, data_state=train_state, bs=opt.batch_size, shuffle=True, data_label=opt.data_label, balanced=opt.balanced_batch, state_args=state_args)
    testloader = get_dataloader3(test_data, data_state=test_state, bs=opt.batch_size, shuffle=False, data_label=opt.data_label, balanced=False, state_args=state_args)
    validloader = get_dataloader3(valid_data, data_state=valid_state, bs=opt.batch_size, shuffle=False, data_label=opt.data_label, balanced=False, state_args=state_args)

    return trainloader, validloader, testloader, additional_info


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    
    model.train()

    total_event_ll = 1  # cumulative event log-likelihood
    total_time_se = 1  # cumulative time prediction squared-error
    total_event_rate = 1  # cumulative number of correct prediction
    total_num_event = 1  # number of total events
    total_num_pred = 1  # number of predictions, total=tqdm_len

    if opt.prof:
        prof =  torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(opt.run_path),
                        record_shapes=True,
                        profile_memory=True,
                        # with_stack=True
                )
        prof.start()


    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):

        

        batch = [x.to(opt.device) for x in batch]

        state_data=[]
        state_label=None
        event_time, time_gap, event_type = batch[:3]
        if opt.state:
            state_time, state_value, state_mod = batch[3:6]
            state_data = batch[3:6]
        if opt.sample_label:
            state_time, state_value, state_mod = batch[3:6]
            state_label=batch[6]
        if opt.demo:
            state_data.append(batch[7])


        # """ prepare data """
        # # event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        # if opt.state or opt.sample_label:
        #     # event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch[0])
        #     # new = next(iter_stateloader)
        #     event_time, time_gap, event_type, state_time, state_value, state_mod, state_label = map(lambda x: x.to(opt.device), batch)
        #     enc_out = model(event_type, event_time, state_time, state_value, state_mod)

        # else:
        #     event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        #     enc_out = model(event_type, event_time)

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        enc_out = model(event_type, event_time, state_data=state_data)

        non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(2)

        

        total_loss = []
        """ backward """
        # negative log-likelihood
        # event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, side = prediction)
        
        
        # if opt.mod != 'None':
        #     if opt.state:
        #         event_ll, non_event_ll = opt.event_loss(model, enc_out, event_time, event_type, side = prediction, mod=opt.mod)
        #     else:
        #         event_ll, non_event_ll = opt.event_loss(model, enc_out, event_time, event_type, mod=opt.mod)
        #     event_loss = -torch.sum(event_ll - non_event_ll)
        # else:
        #     event_loss = torch.zeros((1),device=opt.device)
        

        # CIF decoder
        if hasattr(model, 'event_decoder'):
            log_sum, integral_ = model.event_decoder(enc_out,event_time, event_time, non_pad_mask)
            total_loss.append(  (-torch.sum(log_sum - integral_))  *opt.w_event*1)


        
        # next type prediction
        if hasattr(model, 'pred_next_type'):
            pred_loss, pred_num_event,_ = opt.type_loss(model.y_next_type, event_type, pred_loss_func)
            total_loss.append(pred_loss)

        # next time prediction
        if hasattr(model, 'pred_next_time'):
            non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(-1)
            sse, sse_norm, sae = Utils.time_loss(model.y_next_time, event_time,non_pad_mask) # sse, sse_norm, sae
            total_loss.append(sse*opt.w_time)

        if hasattr(model, 'pred_label'):
            
            # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
            # state_label_loss,_ = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)

            state_label_red = align(state_label[:,:,None], event_time, state_time) # [B,L,1]
            state_label_loss,_ = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask, opt.label_loss_fun)

            total_loss.append(state_label_loss*opt.w_sample_label)

            

    
        loss = sum(total_loss)
        
        # if opt.sample_label:
        #     # state_label_red = align(state_label[:,:,None].float(), event_time, state_time) # [B,L,1]
        #     state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]

        #     state_label_loss,_ = Utils.state_label_loss(state_label_red,prediction[2], non_pad_mask)
        #     loss = loss + state_label_loss*opt.w_sample_label

        # loss=pred_loss*0


        """ forward """
        optimizer.zero_grad()


        loss.backward()

        """ update parameters """
        optimizer.step()

        if opt.prof:
            prof.step()

    if opt.prof:
        prof.stop()
        """ note keeping """
        # total_event_ll += -event_loss.item()
        # total_time_se += sse.item()
        # total_event_rate += pred_num_event.item()
        # total_num_event += non_pad_mask.ne(Constants.PAD).sum().item()
        # # we do not predict the first event
        # if len(event_type.shape)==2:
        #     total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
        # else:
        #     total_num_pred += (event_type+non_pad_mask.unsqueeze(-1)).ne(Constants.PAD).sum().item() - event_time.shape[0]



    # tensorboard [for the last batch]
    # opt.writer.add_scalar("deb3/event_loss", event_loss*opt.w_event,opt.i_epoch )
    # opt.writer.add_scalar("deb3/pred_loss", pred_loss, opt.i_epoch)
    # opt.writer.add_scalar("deb3/sse_loss", sse*opt.w_time, opt.i_epoch)
    # opt.writer.add_scalar("deb3/total_loss", loss, opt.i_epoch)
    # if opt.sample_label:
    #     opt.writer.add_scalar("deb3/state_label_loss", state_label_loss*opt.w_sample_label, opt.i_epoch)


        # prof.step()
    # prof.stop()
    rmse = np.sqrt(total_time_se / total_num_event)

    dict_metrics = {
        # 'f1-micro': metrics.f1_score(y_true, pred_type[~masks].reshape(-1).detach().cpu(),labels= torch.arange(n_classes) ,average='micro', zero_division=0),
        # 'f1-macro': metrics.f1_score(y_true, pred_type[~masks].reshape(-1).detach().cpu(),labels= torch.arange(n_classes) ,average='macro', zero_division=0),
        # 'f1-weighted': metrics.f1_score(y_true, pred_type[~masks].reshape(-1).detach().cpu(),labels= torch.arange(n_classes) ,average='weighted', zero_division=0),
        # 'auc-ovo-macro': metrics.roc_auc_score(y_true, y_score.detach().cpu(), multi_class='ovo',average='macro',labels= torch.arange(n_classes)),
        # 'auc-ovo-weighted': metrics.roc_auc_score(y_true, y_score.detach().cpu(), multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),
        'NLL/#events': -total_event_ll / total_num_event,
        'acc': total_event_rate / total_num_pred,
        'RMSE': rmse,
        # 'auc-ovo-weighted': metrics.roc_auc_score(y_true.detach().cpu(), y_score.detach().cpu(), multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),
        # 'auc-ovo-weighted': metrics.roc_auc_score(y_true.detach().cpu(), y_score.detach().cpu(), multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),

    }

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, dict_metrics


def valid_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """
    example = {}
    model.eval()

    total_event_ll = 1  # cumulative event log-likelihood
    total_event_rate = 1  # cumulative number of correct prediction
    total_num_event = 1  # number of total events
    total_num_pred = 1  # number of predictions, total=tqdm_len
    
    total_time_sse = 1  # cumulative time prediction squared-error
    total_time_sae = 1  # cumulative time prediction squared-error
    total_time_sse_norm = 1  # cumulative time prediction squared-error

    total_label_state = 0  # cumulative time prediction squared-error


    time_gap_true = []
    time_gap_pred = []
    X_enc = []
    y_pred_list = []
    y_true_list = []
    y_score_list = []

    y_state_pred_list = []
    y_state_true_list = []
    y_state_score_list = []
    r_enc_list = []
    masks_list = []

    y_pred_stupid_list = []
    n_classes = model.n_marks

    dict_metrics={}

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                        desc='  - (Testing)   ', leave=False):
            """ prepare data """

            batch = [x.to(opt.device) for x in batch]
            # batch = map(lambda x: x.to(opt.device), batch)

            state_data=[]
            state_label=None
            # if opt.event_enc:
            event_time, time_gap, event_type = batch[:3]
            if opt.state:
                state_time, state_value, state_mod = batch[3:6]
                state_data = batch[3:6]
            if opt.sample_label:
                state_time, state_value, state_mod = batch[3:6]
                state_label=batch[6]
            if opt.demo:
                state_data.append(batch[7])
            
            # if opt.state or opt.sample_label:
            #     # event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch[0])
            #     # new = next(iter_stateloader)
            #     event_time, time_gap, event_type, state_time, state_value, state_mod, state_label = map(lambda x: x.to(opt.device), batch)
            #     enc_out = model(event_type, event_time, state_time, state_value, state_mod)

            # else:
            #     event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
            #     enc_out = model(event_type, event_time)

            # r_enc_list.append(prediction[2][:,1:,:])
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            enc_out = model(event_type, event_time, state_data=state_data)
            # total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(2)
            total_num_pred += non_pad_mask.sum().item()

            # CIF decoder
            if hasattr(model, 'event_decoder'):
                log_sum, integral_ = model.event_decoder(enc_out,event_time, event_time, non_pad_mask)
                # total_loss.append(  (-torch.sum(log_sum - integral_))  *opt.w_event*1)
                total_event_ll += torch.sum(log_sum - integral_)

            # next type prediction
            if hasattr(model, 'pred_next_type'):
                pred_loss, pred_num_event,(y_pred, y_true, y_score, masks) = opt.type_loss(model.y_next_type, event_type, pred_loss_func)
                # total_loss.append(pred_loss)
                y_pred_list.append( torch.flatten(y_pred, end_dim=1) ) # [*]
                y_true_list.append( torch.flatten(y_true, end_dim=1) ) # [*]
                y_score_list.append( torch.flatten(y_score, end_dim=1) ) # [*, C]
            masks_list.append( non_pad_mask[:,1:].flatten().bool() ) # [*, C]

            # next time prediction
            if hasattr(model, 'pred_next_time'):
                non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(-1)
                sse, sse_norm, sae = Utils.time_loss(model.y_next_time, event_time,non_pad_mask) # sse, sse_norm, sae
                # total_loss.append(sse*opt.w_time)
                total_time_sse += sse.item()  # cumulative time prediction squared-error
                total_time_sae += sae.item()  # cumulative time prediction squared-error
                total_time_sse_norm += sse_norm.item()  # cumulative time prediction squared-error

            # label prediction
            if hasattr(model, 'pred_label') and (state_label is not None):
                
                # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
                # state_label_loss,(y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)
                
                state_label_red = align(state_label[:,:,None], event_time, state_time) # [B,L,1]
                # state_label_red = state_label.bool().int()[:,:,None] # [B,L,1]
                state_label_loss,(y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask,opt.label_loss_fun)

                

                # total_loss.append(state_label_loss*opt.w_sample_label)
                y_state_pred_list.append(  torch.flatten(y_state_pred) ) # [*]
                y_state_true_list.append(  torch.flatten(y_state_true) ) # [*]
                y_state_score_list.append(  torch.flatten(y_state_score) ) # [*] it is binary

    masks = torch.cat(masks_list) # [*]


    # CIF decoder
    if hasattr(model, 'event_decoder'):
        # log_sum, integral_ = model.event_decoder(enc_out,event_time, event_time, non_pad_mask)
        # total_loss.append(  (-torch.sum(log_sum - integral_))  *opt.w_event*1)

        
        dict_metrics.update({
            'CIF/NLL-#events': -total_event_ll.item() / total_num_pred,
            'CIF/NLL': -total_event_ll.item(),
            'CIF/#events': total_num_pred,
        })

    # next time prediction
    if hasattr(model, 'pred_next_time'):
        rmse = np.sqrt(total_time_sse / total_num_pred)
        msae = total_time_sae / total_num_pred
        rmse_norm = np.sqrt(total_time_sse_norm / total_num_pred)
        dict_metrics.update({
                'NextTime/RMSE': rmse,
                'NextTime/rmse_norm': rmse_norm,
                'NextTime/msae': msae,
        })
    


    

    # next type prediction
    if hasattr(model, 'pred_next_type'):
        # pred_loss, pred_num_event,(y_pred, y_true, y_score, masks) = opt.type_loss(model.y_next_type, event_type, pred_loss_func)
        # total_loss.append(pred_loss)
        # y_pred_list.append( torch.flatten(y_pred, end_dim=1) ) # [*]
        # y_true_list.append( torch.flatten(y_true, end_dim=1) ) # [*]
        # y_score_list.append( torch.flatten(y_score, end_dim=1) ) # [*, C]
        # masks_list.append( non_pad_mask[:,1:].flatten().bool() ) # [*, C]


        if y_pred_list[-1].dim()==2: # multilabel or marked
            y_pred = (            torch.cat(y_pred_list) [  masks, :   ]                      ).detach().cpu()
            y_true = (            torch.cat(y_true_list) [  masks, :   ]                      ).detach().cpu()
            y_score = (           torch.cat(y_score_list) [  masks, :   ]                        ).detach().cpu()

            cm = metrics.multilabel_confusion_matrix(y_true, y_pred)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)

            dict_metrics.update({

                'NextType(ML)/auc-ovo-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='weighted'),
                'NextType(ML)/auc-PR-weighted': metrics.average_precision_score(y_true, y_score ,average='weighted'),
                'NextType(ML)/f1-weighted': metrics.f1_score(y_true, y_pred ,average='weighted', zero_division=0),
                'NextType(ML)/precision-weighted': metrics.precision_score(y_true, y_pred ,average='weighted', zero_division=0),
                'NextType(ML)/recall-weighted': metrics.recall_score(y_true, y_pred ,average='weighted', zero_division=0),
                
                # 'NextType(ML)/f1-micro': metrics.f1_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/f1-macro': metrics.f1_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/f1-samples': metrics.f1_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/acc': metrics.accuracy_score(y_true, y_pred),
                # 'NextType(ML)/hamming': metrics.hamming_loss(y_true, y_pred),

                # 'NextType(ML)/precision-micro': metrics.precision_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/precision-macro': metrics.precision_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/precision-samples': metrics.precision_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/recall-micro': metrics.recall_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/recall-macro': metrics.recall_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/recall-samples': metrics.recall_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro'),
                # 'NextType(ML)/auc-ovo-micro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='micro'),
                # 'NextType(ML)/auc-PR-micro': metrics.average_precision_score(y_true, y_score ,average='micro'),
                # 'NextType(ML)/auc-PR-macro': metrics.average_precision_score(y_true, y_score ,average='macro'),

                # 'MultiLabel/AUROC-ovo-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='weighted'),
                # 'MultiLabel/AUPRC-weighted': metrics.average_precision_score(y_true, y_score ,average='weighted'),
                # 'MultiLabel/F1-weighted': metrics.f1_score(y_true, y_pred ,average='weighted', zero_division=0),

                # 'ConfMat': cm_display,

            })
        else:   # multiclass
            y_pred = (            torch.cat(y_pred_list)[masks]                      ).detach().cpu()
            y_true = (            torch.cat(y_true_list)[masks]                   ).detach().cpu()
            y_score = (           torch.cat(y_score_list)[masks,: ]                      ).detach().cpu()

            cm = metrics.confusion_matrix(y_true, y_pred)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
            dict_metrics.update({
                # 'NextType(MC)/f1-micro': metrics.f1_score(y_true, y_pred, labels= torch.arange(n_classes) ,average='micro', zero_division=0),
                # 'NextType(MC)/f1-macro': metrics.f1_score(y_true, y_pred, labels= torch.arange(n_classes) ,average='macro', zero_division=0),
                'NextType(MC)/f1-weighted': metrics.f1_score(y_true, y_pred, labels= torch.arange(n_classes) ,average='weighted', zero_division=0),

                'NextType(MC)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro',labels= torch.arange(n_classes)),
                'NextType(MC)/auc-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),
                # 'auc-ovo-weighted-stupid': metrics.roc_auc_score(y_true, y_pred_stupid, multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),



                'NextType(MC)/acc': metrics.accuracy_score(y_true, y_pred, normalize=True),
                # 'acc_stupid': metrics.accuracy_score(y_true, y_pred_stupid, normalize=True),


                # 'MAE':mae,
                # 'MNAE':mnae,

                # 'acc_old': total_event_rate / total_num_pred,

                # 'RMSE-stupid': np.sqrt(np.mean((time_gap_true-time_gap_pred_stupid)**2)),
                # 'MAE-stupid': np.mean(np.absolute(time_gap_true-time_gap_pred_stupid)),
                # 'MNAE-stupid': np.mean(np.absolute(time_gap_true-time_gap_pred_stupid)) / np.mean(time_gap_true),

                'ConfMat': cm_display,
                # 'time_gap_data': [time_gap_true, time_gap_pred],
                # 'tsne':{'X_enc':X_enc,'X_tsne':X_tsne,'y_true':y_true, 'y_pred':y_pred}

            })

    # label prediction
    if hasattr(model, 'pred_label'):
        
        # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
        # state_label_loss,(y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)
        # # total_loss.append(state_label_loss*opt.w_sample_label)
        # y_state_pred_list.append(y_state_pred) # [*]
        # y_state_true_list.append(y_state_true) # [*]
        # y_state_score_list.append(y_state_score) # [*] it is binary
        
        # y_state_pred = (torch.cat(y_state_pred_list)).detach().cpu() # [*]
        # y_state_true = (torch.cat(y_state_true_list)).detach().cpu()
        # y_state_score = (torch.cat(y_state_score_list)).detach().cpu()

        # y_state_pred = (torch.cat(y_state_pred_list) [masks]).detach().cpu() # [*]
        # y_state_true = (torch.cat(y_state_true_list) [masks]).detach().cpu()
        # y_state_score = (torch.cat(y_state_score_list) [masks]).detach().cpu()

        y_state_pred = (torch.cat(y_state_pred_list)).detach().cpu() # [*]
        y_state_true = (torch.cat(y_state_true_list)).detach().cpu()
        y_state_score = (torch.cat(y_state_score_list)).detach().cpu()

        dict_metrics.update({
            'pred_label/AUROC': metrics.roc_auc_score(y_state_true, y_state_score),
            'pred_label/AUPRC': metrics.average_precision_score(y_state_true, y_state_score),
            'pred_label/f1-binary': metrics.f1_score(y_state_true, y_state_pred ,average='binary', zero_division=0),

            'pred_label/loss':total_label_state/total_num_event,
            'pred_label/recall-binary': metrics.recall_score(y_state_true, y_state_pred ,average='binary', zero_division=0),
            'pred_label/precision-binary': metrics.precision_score(y_state_true, y_state_pred ,average='binary', zero_division=0),

            'pred_label/ACC': metrics.accuracy_score(y_state_true, y_state_pred),


            # 'pred_label/sum_r_enc':torch.concat(r_enc_list, dim=1).sum().item(),
            # 'pred_label/r_enc_zero': a,
        })
  







    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, dict_metrics


def train(model, trainloader, validloader, testloader, optimizer, scheduler, pred_loss_func, opt, trial=None):
    """ Start training. """
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE


    # PRE eval
    start = time.time()
    valid_event, valid_type, valid_time, dict_metrics_test = valid_epoch(model, validloader, pred_loss_func, opt)
    print('  - (PRE Testing)     loglikelihood: {ll: 8.5f}, '
            'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

    # logging
    write_to_summary(dict_metrics_test, opt, i_epoch=0,prefix='Valid-')

    best_metric=0
        # Initialize the early stopping counter
    early_stopping_counter = 0

    # Set the maximum number of epochs without improvement
    max_epochs_without_improvement = opt.ES_pat

    dict_time={}


    best_test_metric = {}
    best_valid_metric = {}

    if opt.sample_label:
        best_test_metric.update({'pred_label/f1-binary':0, 'pred_label/AUPRC':0, 'pred_label/AUROC':0})
        best_valid_metric.update({'pred_label/f1-binary':0, 'pred_label/AUPRC':0, 'pred_label/AUROC':0})

    for epoch_i in tqdm(range(opt.epoch), leave=False):
        epoch = epoch_i + 1
        # print('[ Epoch', epoch, ']')
        opt.i_epoch = epoch


        # ********************************************* Train Epoch *********************************************
        start = time.time()
        train_event, train_type, train_time, dict_metrics_train = train_epoch(model, trainloader, optimizer, pred_loss_func, opt)
        # print('  - (Training)    loglikelihood: {ll: 8.5f}, '
        #       'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
        #       'elapse: {elapse:3.3f} min'
        #       .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        dict_time.update({'Time/train_epoch':((time.time() - start) / 60)})
        for k,v in dict_metrics_train.items():
            opt.writer.add_scalar('Trainn/'+k, v, epoch_i)
        
        
        # Train eval *********************************************
        train_event, train_type, train_time, dict_metrics_train = valid_epoch(model, trainloader, pred_loss_func, opt)

        write_to_summary(dict_metrics_train, opt, i_epoch=epoch, prefix='Train-')

        # ********************************************* Valid Epoch *********************************************
        start = time.time()
        valid_event, valid_type, valid_time, dict_metrics_valid = valid_epoch(model, validloader, pred_loss_func, opt)
        # print('  - (Validating)     loglikelihood: {ll: 8.5f}, '
        #       'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
        #       'elapse: {elapse:3.3f} min'
        #       .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        # print('  - [Info] Maximum ll: {event: 8.5f}, '
        #       'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
        #       .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        dict_time.update({'Time/valid_epoch':((time.time() - start) / 60)})

        write_to_summary(dict_metrics_valid, opt, i_epoch=epoch, prefix='Valid-')


        # ********************************************* Test Epoch *********************************************
        if testloader is not None:
            start = time.time()
            test_event, test_type, test_time, dict_metrics_test = valid_epoch(model, testloader, pred_loss_func, opt)
           


            write_to_summary(dict_metrics_test, opt, i_epoch=epoch, prefix='Test-')




        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        
        

        write_to_summary(dict_time, opt, i_epoch=epoch, prefix='time-')


        

        scheduler.step()

        if opt.i_epoch % 5==0:
            torch.save({
                'epoch': opt.i_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dict_metrics_test':dict_metrics_test,
            }, opt.run_path+'model_ep'+str(opt.i_epoch)+'.pkl')



        if 'pred_label/f1-binary' in dict_metrics_test: 
            inter_Obj_val=dict_metrics_test['pred_label/f1-binary']
        elif 'NextType(ML)/f1-weighted' in dict_metrics_test: 
            inter_Obj_val=dict_metrics_test['NextType(ML)/f1-weighted']
        elif 'NextType(MC)/f1-weighted' in dict_metrics_test:
            inter_Obj_val=dict_metrics_test['NextType(MC)/f1-weighted']
        elif 'pred_label/AUPRC' in dict_metrics_test:
            inter_Obj_val=dict_metrics_test['pred_label/AUPRC']   
        else:
            inter_Obj_val=0
        
        # inter_Obj_val=best_valid_metric['pred_label/f1-binary']
        
        opt.writer.add_scalar('Obj', inter_Obj_val, global_step=opt.i_epoch)
        if opt.wandb:
            wandb.log({'Obj': inter_Obj_val}, step=opt.i_epoch)


        # Early stopping

        for k,v in best_valid_metric.items():
            if dict_metrics_valid[k]>v:
                best_valid_metric[k]= dict_metrics_valid[k]
                best_test_metric[k]= dict_metrics_test[k]
            if opt.wandb:
                wandb.log({('Best-Test-'+k):v for k,v in best_test_metric.items()}, step=opt.i_epoch)
                wandb.log({('Best-Valid-'+k):v for k,v in best_valid_metric.items()}, step=opt.i_epoch)
        if inter_Obj_val > best_metric:
            # Save the model weights
            # torch.save(model.state_dict(), "best_model.pth")
            
            # Reset the early stopping counter
            early_stopping_counter = 0
            
            # Update the best metric
            best_metric = inter_Obj_val
            
        else:
            # Increment the early stopping counter
            early_stopping_counter += 1
            
            # Check if the early stopping counter has reached the maximum number of epochs without improvement
            if early_stopping_counter >= max_epochs_without_improvement:
                print("Early stopping at epoch {}".format(epoch))
                if opt.wandb:
                    wandb.run.summary["max_obj_val"] = best_metric
                    wandb.run.summary["status"] = "stopped"
                    # wandb.finish(quiet=True)
                break





        # Pruning
        if trial is not None:
            # raise optuna.TrialPruned()

             # Handle pruning based on the intermediate value.            
            trial.report(inter_Obj_val, opt.i_epoch)            
            if trial.should_prune():
                
                opt.writer.add_hparams(opt.hparams2write, {'Objective':inter_Obj_val},run_name='../'+opt.run_name)
                return best_metric

                # opt.writer.add_hparams(opt.hparams2write, {'Objective':inter_Obj_val},run_name='../'+opt.run_name)
                # if opt.wandb:
                    
                #     wandb.run.summary["status"] = "pruned"
                #     wandb.finish(quiet=True)
                #     opt.wandb=False
                #     return best_metric


                # raise optuna.exceptions.TrialPruned()
    opt.writer.add_hparams(opt.hparams2write, {'Objective':inter_Obj_val},run_name='../'+opt.run_name)
    return best_metric

def options():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('-data', default="C:/DATA/data/processed/physio2019_1d_HP_std/", required=False)
    parser.add_argument('-data_label',choices=['multiclass','multilabel'], default='multilabel')

    parser.add_argument('-cuda', type=int, choices=[0,1], default=1, help='consider cuda?')

    parser.add_argument('-wandb', action='store_true', dest='wandb', help='consider wandb?')
    parser.add_argument('-prof', action='store_true', dest='prof', help='consider profiling?')

    parser.add_argument('-per', type=int, default=25, help='percentage of dataset to be used for training')
    parser.add_argument('-unbalanced_batch', action='store_false', dest='balanced_batch', help='not balanced mini batches?')

    parser.add_argument('-transfer_learning', default="", help='specify run_name')
    parser.add_argument('-freeze', choices=['TEDA','TE','DA',''], default="", help='specify run_name')

    parser.add_argument('-ES_pat', type=int, default=10, help='max_epochs_without_improvement')
    


    # data handling


    parser.add_argument('-setting', type=str,choices=['sc','mc1','mc2',''], default='', help='max_epochs_without_improvement')
    parser.add_argument('-test_center', type=str, default='', help='max_epochs_without_improvement')
    parser.add_argument('-split', type=str, default='', help='max_epochs_without_improvement')


    # logging args
    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-user_prefix', type=str, default='SYN')
    parser.add_argument('-pos_alpha', type=float, default=1.0)


    # data handling
    # parser.add_argument('-data_setting', choices=['sc','mc1','mc2',''], default="", help='settings')
    # parser.add_argument('-test_center', type=str, default='')
    # parser.add_argument('-test_split', type=str, default='')
    # parser.add_argument('-train_center', type=str, default='')

    # General Config
    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-lr', type=float, default=3e-4)
    parser.add_argument('-smooth', type=float, default=0.0)
    parser.add_argument('-weight_decay', type=float, default=1e-0)


    # Transformer Encoder Architecture
    parser.add_argument('-event_enc', type=int, choices=[0,1], default=1, help='consider state?')

    parser.add_argument('-time_enc', type=str, choices=['sum','concat','none'], default='concat', help='specify time encoding')
        # hyper paramters
    parser.add_argument('-te_d_mark', type=int, default=8)
    parser.add_argument('-te_d_time', type=int, default=8)

    parser.add_argument('-te_d_rnn', type=int, default=256)
    parser.add_argument('-te_d_inner', type=int, default=16)
    parser.add_argument('-te_d_k', type=int, default=8)
    parser.add_argument('-te_d_v', type=int, default=8)
    parser.add_argument('-te_n_head', type=int, default=4)
    parser.add_argument('-te_n_layers', type=int, default=4)
    parser.add_argument('-te_dropout', type=float, default=0.1)


    # State Encoder Architecture
    parser.add_argument('-state', action='store_true', dest='state', help='consider state?')
    parser.add_argument('-demo', action='store_true', dest='demo', help='consider demo?')

    parser.add_argument('-num_states', type=int, default=1)
        # hyper paramters


    # Outputs


    parser.add_argument('-w_sample_label', type=int, default=100.0)

        # CIFs
    parser.add_argument('-mod', type=str, choices=['single','mc','ml','none'], default='single', help='specify the mod')
    parser.add_argument('-int_dec', type=str, choices=['thp','sahp'], default='sahp', help='specify the inteesity decoder')
    parser.add_argument('-w_event', type=int, default=1)

        # marks
    parser.add_argument('-next_mark',  type=int, choices=[0,1], default=1, help='0: mark not detached, 1: mark detached')
    parser.add_argument('-w_class', action='store_true', dest='w_class', help='consider w_class?')
    parser.add_argument('-w_pos', action='store_true', dest='w_pos', help='consider w_pos?')
    # parser.add_argument('-mark_detach', action='store_true', dest='w_pos', help='consider w_pos?')
    parser.add_argument('-mark_detach',  type=int, choices=[0,1], default=0, help='0: mark not detached, 1: mark detached')

        # times
    parser.add_argument('-w_time', type=int, default=1.0)

        # final sample label
    # parser.add_argument('-sample_label', action='store_true', dest='sample_label', help='consider state?')
    parser.add_argument('-sample_label',  type=int, choices=[0,1,2], default=0, help='2 for detach')
    parser.add_argument('-w_pos_label', type=float, default=1.0)



    opt = parser.parse_args()

    temp = vars(opt)

    opt.hparams2write = {}
    for k,v in temp.items():
        if type(v) in [str,int,float,bool,torch.tensor]:
            opt.hparams2write[k] = v

    return opt

def config(opt, justLoad=False):
    if justLoad is False:

        # it is a new run
        t0 = datetime.datetime.strptime("17-9-22--00-00-00", "%d-%m-%y--%H-%M-%S")
        t_now = datetime.datetime.now()
        t_diff = t_now-t0
        opt.date = time.strftime("%d-%m-%y--%H-%M-%S")
        opt.run_id = str(t_diff.days)+str(int(t_diff.seconds/10))


        print(f"[Info] ### Point Process strategy: {opt.mod} ###")

        if opt.setting=='':
            opt.str_config = '-'
            # Tensorboard integration
            opt.run_name = opt.user_prefix+str(opt.run_id)+opt.str_config
            opt.run_path = opt.data + opt.run_name+'/'
        else:
            if opt.setting=='mc2':
                opt.str_config = '-'+opt.setting+'-H'+opt.test_center
            else:
                opt.str_config = '-'+opt.setting+'-H'+opt.test_center+'/split'+opt.split
            # Tensorboard integration
            opt.run_name = opt.user_prefix+str(opt.run_id)
            opt.run_path = opt.data[:-1]+opt.str_config+'/'+ opt.run_name+'/'
            
            opt.dataset = opt.data
            opt.data=opt.data[:-1]+opt.str_config+'/'
        # create a foler for the run
        

        if os.path.exists(opt.run_path):
            # print(settings.load_model)
            shutil.rmtree(opt.run_path)
        os.makedirs(opt.run_path, exist_ok=True)


        with open(opt.run_path+'opt.pkl','wb') as f:
            pickle.dump(opt,f)


    


    if opt.transfer_learning != '':

        opt.all_transfered_modules=['pred_next_time']
        # if opt.sample_label:
        #     opt.all_transfered_modules.append('pred_label')
        if opt.next_mark:
            opt.all_transfered_modules.append('pred_next_type')
        if opt.mod != 'none':
            opt.all_transfered_modules.append('event_decoder')
        if opt.state:
            opt.all_transfered_modules.append('DAM')
        if opt.event_enc:
            opt.all_transfered_modules.append('TE')

        opt.freezed_modules = []
        if opt.freeze=='DA':
            opt.freezed_modules.append('DAM')
        if opt.freeze=='TE':
            opt.freezed_modules.append('TE')

    
    # Event loss handling
    # if opt.int_dec=='thp':
    #     opt.event_loss = Utils.thp_log_likelihood
    #     opt.event_loss_test = Utils.thp_log_likelihood_test

    #     # if opt.state:
    #     #     opt.event_loss = Utils.sahp_state_log_likelihood
    #     #     opt.event_loss_test = Utils.sahp_state_log_likelihood_test
    #     # else:
    #     #     opt.event_loss = Utils.sahp_log_likelihood
    #     #     opt.event_loss_test = Utils.sahp_log_likelihood_test

    # elif opt.int_dec=='sahp':

    #     if opt.state:
    #         opt.event_loss = Utils.sahp_state_log_likelihood
    #         opt.event_loss_test = Utils.sahp_state_log_likelihood_test
    #     else:
    #         # opt.event_loss = Utils.sahp_log_likelihood
    #         # opt.event_loss_test = Utils.sahp_log_likelihood_test
    #         opt.event_loss = Utils.sahp_state_log_likelihood
    #         opt.event_loss_test = Utils.sahp_state_log_likelihood_test


    opt.device = torch.device('cuda') if (torch.cuda.is_available() and opt.cuda) else torch.device('cpu')

    # setup the log file
    # with open(opt.log, 'w') as f:
    #     f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    # if justLoad is False:
    opt.trainloader, opt.validloader, opt.testloader, additional_info = prepare_dataloader(opt)

    if opt.mod=='single' or opt.mod=='None':
        opt.num_types=1
        opt.num_types = additional_info['num_types']
    elif opt.mod=='mc':
        opt.num_types = additional_info['num_types']
    elif opt.mod=='ml':
        opt.num_types = additional_info['num_marks']
        

    

    if 'num_marks' in additional_info:
        opt.num_marks = additional_info['num_marks']
    else:
        opt.num_marks = opt.num_types
    if 'num_states' in additional_info:
        opt.num_states = additional_info['num_states']
    if 'pos_weight' in additional_info:
        opt.pos_weight = additional_info['pos_weight']

    if 'w' in additional_info:
        opt.w = additional_info['w']
    if 'num_marks' in additional_info:
        opt.num_marks = additional_info['num_marks']

    if 'num_demos' in additional_info:
        opt.num_demos = additional_info['num_demos']
    else:
        opt.num_demos=0

    if opt.w_class:
        # opt.w_class = [0.01602763, 0.01989574, 0.0247974 , 0.04106871, 0.02056413,
        #                     0.01968509, 0.02592123, 0.03442   , 0.02894503, 0.03738965,
        #                     0.03015046, 0.03117576, 0.03434853, 0.03477166, 0.04845101,
        #                     0.03590284, 0.05527879, 0.05201981, 0.13104338, 0.12973979,
        #                     0.14840336]
        w = torch.tensor(opt.w, device=opt.device)
        print('[Info] class weigths:\n',w)

    else:
        w = torch.ones(opt.num_marks, device=opt.device)/opt.num_marks

    if opt.w_pos:
        opt.pos_weight = torch.tensor(opt.pos_weight, device=opt.device)
        opt.pos_weight = opt.pos_weight*opt.pos_alpha
        print('[Info] pos weigths:\n',opt.pos_weight)

    else:
        opt.pos_weight = torch.ones(opt.num_marks, device=opt.device)


    if opt.w_class:
        opt.w = torch.tensor(opt.w, device=opt.device)
        print('[Info] pos weigths:\n',opt.w)

    else:
        opt.w = torch.ones(opt.num_marks, device=opt.device)

    # if opt.mod=='SHP_marked' or opt.mod=='None':
    #     opt.type_loss = Utils.type_loss_BCE
    #     opt.pred_loss_func = nn.BCEWithLogitsLoss(reduction='none', weight=opt.w, pos_weight=opt.pos_weight)
    # elif opt.mod=='MHP_multiclass':
    #     opt.type_loss = Utils.type_loss_CE
    #     opt.pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', weight=w)
    # elif opt.mod=='MHP_multilabel':
    #     opt.type_loss = Utils.type_loss_BCE
    #     opt.pred_loss_func = nn.BCEWithLogitsLoss(reduction='none', weight=opt.w, pos_weight=opt.pos_weight)


    if opt.data_label=='multilabel':
        opt.type_loss = Utils.type_loss_BCE
        opt.pred_loss_func = nn.BCEWithLogitsLoss(reduction='none', weight=opt.w, pos_weight=opt.pos_weight)
    elif opt.data_label=='multiclass':
        opt.type_loss = Utils.type_loss_CE
        opt.pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', weight=w)
    
    opt.label_loss_fun = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(opt.w_pos_label, device=opt.device))

    opt.TE_config={}
    if opt.event_enc:
        opt.TE_config['n_marks'] = opt.num_marks              
        opt.TE_config['d_type_emb'] = opt.te_d_mark

        opt.TE_config['time_enc'] = opt.time_enc
        opt.TE_config['d_time'] = opt.te_d_time
        
        opt.TE_config['d_inner'] = opt.te_d_inner
        opt.TE_config['n_layers'] = opt.te_n_layers
        opt.TE_config['n_head'] = opt.te_n_head
        opt.TE_config['d_k'] = opt.te_d_k
        opt.TE_config['d_v'] = opt.te_d_v
        opt.TE_config['dropout'] = opt.te_dropout



    opt.DAM_config = {}
    if opt.state:

        opt.DAM_config['output_activation'] = 'relu'
        opt.DAM_config['output_dims'] = 4
                    
                    # MLP encoder for combined values
        opt.DAM_config['n_phi_layers'] = 3
        opt.DAM_config['phi_width'] = 32
        opt.DAM_config['phi_dropout'] = 0
                    
                    # Cumulative Set Attention Layer
        opt.DAM_config['n_psi_layers'] = 2
        opt.DAM_config['psi_width'] = 16
        opt.DAM_config['psi_latent_width'] = 32
                    
        opt.DAM_config['dot_prod_dim'] = 16
        opt.DAM_config['n_heads'] = 2
        opt.DAM_config['attn_dropout'] = 0
        opt.DAM_config['latent_width'] = 16

        
        opt.DAM_config['n_rho_layers'] = 2
        opt.DAM_config['rho_width'] = 32 #256
        opt.DAM_config['rho_dropout'] = 0

        opt.DAM_config['max_timescale'] = 1000
        opt.DAM_config['n_positional_dims'] = 4
        opt.DAM_config ['num_mods'] = opt.num_states
        opt.DAM_config ['num_demos'] = opt.num_demos
        opt.DAM_config ['online'] = False








    opt.demo_config={}
    if opt.demo:
        opt.demo_config['num_demos']=additional_info['num_demos']
        opt.demo_config['d_demo']=4

    opt.CIF_config = {}
    if opt.mod != 'none':
        opt.CIF_config['mod']=opt.mod
        opt.CIF_config['type']=opt.int_dec
    
        if opt.CIF_config['mod']=='single':
            opt.CIF_config['n_cifs']=1
        else:
            opt.CIF_config['n_cifs']=opt.num_marks


    opt.next_type_config = {}
    if opt.next_mark:
        opt.next_type_config['n_marks'] = opt.num_marks 
    opt.next_time_config = True
    opt.label_config = opt.sample_label

    return opt


def process_hparams(trial,opt):

    # argv=['Main.py']
    # for hparam in hparams:
    #     # argv.extend([k,str(v)])
    #     if hparam != '':
    #         argv.append(str(hparam))

    # print(argv)

    # opt.w_pos_label = trial.suggest_float('w_pos_label',0.2, 5,step=0.1)

    opt.weight_decay = trial.suggest_float('weight_decay',1e-4, 1,log=True)

    opt.lr = trial.suggest_float('lr',1e-5, 1e-2,log=True)
    # opt.batch_size = trial.suggest_categorical('batch_size', [8,16,32,64,128])

    if opt.event_enc:
        # # # opt.TE_config['n_marks'] = opt.num_marks              
        # opt.TE_config['d_type_emb'] = trial.suggest_categorical('d_type_emb', [8,16])

        # # opt.TE_config['time_enc'] = trial.suggest_categorical('time_enc', ['sum','concat','none'])
        # opt.TE_config['d_time'] = trial.suggest_categorical('d_time', [8,16])
        
        # opt.TE_config['d_inner'] = trial.suggest_categorical('d_inner', [8,16])
        # opt.TE_config['n_layers'] = trial.suggest_categorical('n_layers', [2,4,8])
        # opt.TE_config['n_head'] = trial.suggest_categorical('n_head', [2,4,8])
        # opt.TE_config['d_k'] = trial.suggest_categorical('d_k', [8])
        # opt.TE_config['d_v'] = trial.suggest_categorical('d_v', [8])
        # opt.TE_config['dropout'] = trial.suggest_categorical('dropout', [8,16])
        aaa=1



    if opt.state:

        # opt.DAM_config['output_activation'] = trial.suggest_categorical('output_activation', [8,16])
        # opt.DAM_config['output_dims'] = trial.suggest_categorical('output_dims', [8,16])
                    
        #             # MLP encoder for combined values
        # opt.DAM_config['n_phi_layers'] = trial.suggest_categorical('n_phi_layers', [8,16])
        # opt.DAM_config['phi_width'] = trial.suggest_categorical('phi_width', [8,16])
        # opt.DAM_config['phi_dropout'] = trial.suggest_categorical('phi_dropout', [8,16])
                    
        #             # Cumulative Set Attention Layer
        # opt.DAM_config['n_psi_layers'] = trial.suggest_categorical('n_psi_layers', [8,16])
        # opt.DAM_config['psi_width'] = trial.suggest_categorical('psi_width', [8,16])
        # opt.DAM_config['psi_latent_width'] = trial.suggest_categorical('psi_latent_width', [8,16])
                    
        opt.DAM_config['dot_prod_dim'] = 16 # trial.suggest_categorical('dot_prod_dim', [16,32,64])
        opt.DAM_config['n_heads'] = 2 #trial.suggest_categorical('n_heads', [2,4])
        # opt.DAM_config['attn_dropout'] = trial.suggest_categorical('attn_dropout', [8,16])
        opt.DAM_config['latent_width'] = 16 #trial.suggest_categorical('latent_width', [16,128])

        
        # opt.DAM_config['n_rho_layers'] = trial.suggest_categorical('n_rho_layers', [8,16])
        opt.DAM_config['rho_width'] = 256 #trial.suggest_categorical('rho_width', [32,256])
        # opt.DAM_config['rho_dropout'] = trial.suggest_categorical('rho_dropout', [8,16])

        # opt.DAM_config['max_timescale'] = trial.suggest_categorical('max_timescale', [8,16])
        # opt.DAM_config['n_positional_dims'] = trial.suggest_categorical('n_positional_dims', [8,16])
        # # # opt.DAM_config ['num_mods'] = opt.num_states
        # # # opt.DAM_config ['num_demos'] = opt.num_demos
        # # # opt.DAM_config ['online'] = False

        aa=1







    # opt.demo_config={}
    # if opt.demo:
    #     opt.demo_config['num_demos']=additional_info['num_demos']
    #     opt.demo_config['d_demo']=4

    # opt.CIF_config = {}
    # if opt.mod != 'none':
    #     opt.CIF_config['mod']=opt.mod
    #     opt.CIF_config['type']=opt.int_dec
    
    #     if opt.CIF_config['mod']=='single':
    #         opt.CIF_config['n_cifs']=1
    #     else:
    #         opt.CIF_config['n_cifs']=opt.num_marks


    # opt.next_type_config = {}
    # if opt.next_mark:
    #     opt.next_type_config['n_marks'] = opt.num_marks 
    # opt.next_time_config = True
    # opt.label_config = opt.sample_label

    config_hparams = dict(trial.params)
    config_hparams["trial.number"] = trial.number

    return opt, config_hparams


def load_module(model, checkpoint, modules, to_freeze=True):



    for module in modules:

        b=[x for x in checkpoint['model_state_dict'].keys() if x.startswith(module)]
        od = OrderedDict()
        for k in b:
            od[k[  (len(module)+1)  :]]=checkpoint['model_state_dict'][k]

        # model.encoder.load_state_dict(od)
        getattr(model,module).load_state_dict(od)
        for para in getattr(model,module).parameters():
            para.requires_grad = not to_freeze

    return



def main(trial=None):
    """ Main function. """
    

    print(sys.argv)

    opt=options() # if run from command line it process sys.argv


    


    

    if opt.transfer_learning != '':
        # add_data = "C:/DATA/data/processed/physio2019_1d_HP_std/"
        TL_run_add = opt.data+opt.transfer_learning
        old_freeze = opt.freeze

        old_user_prefix = opt.user_prefix
        temp = opt.transfer_learning


        # find all files in the directory
        os.chdir(opt.data)
        candidateFiles = filter(lambda x:  (opt.transfer_learning in x), os.listdir(opt.data))
        candidateFiles = [f for f in candidateFiles]
        candidateFiles.sort(key=lambda x: os.path.getmtime(x)) # newest last

        if len(candidateFiles)==0:
            raise Exception('### No candidateFiles')
        # elif len(candidateFiles)>1:
        #      raise Exception('### Many candidateFiles')
        else:
            TL_run=candidateFiles[-1]

        print('### ---------TRANSFER LEARNING----------->    '+TL_run)

        # load opt file
        with open(opt.data+TL_run+'/opt.pkl','rb') as f:
            opt = pickle.load(f)
        

        # all_transfered_modules=['pred_next_time']
        # # if opt.sample_label:
        # #     all_transfered_modules.append('pred_label')
        # if opt.next_mark:
        #     all_transfered_modules.append('pred_next_type')
        # if opt.mod != 'none':
        #     all_transfered_modules.append('event_decoder')
        # if opt.state:
        #     all_transfered_modules.append('DAM')
        # if opt.event_enc:
        #     all_transfered_modules.append('TE')

        opt.user_prefix = old_user_prefix

        # override
        # opt.mod = 'None'
        # opt.mark_detach = 1
        opt.sample_label = 1
        
        opt.transfer_learning=temp
        opt.freeze = old_freeze
        

    opt = config(opt)
    
    
    if isinstance(trial,optuna.trial._trial.Trial):
        print('OPTUNA!')
        opt, config_hparams = process_hparams(trial, opt)
        opt.hparams = config_hparams
        if opt.wandb:
            wandb.login()
            # wandb.tensorboard.patch(root_logdir=opt.run_path, pytorch=True)
            # sync_tensorboard=True,
            wandb.init( config=opt,
                        project="TEDAM3",
                        entity="hokarami",
                        group=opt.user_prefix,
                        name=opt.run_name,
                        reinit=True,
                        # settings=wandb.Settings(start_method="fork")
                    )
            # wandb.config.update(opt.TE_config)
            # wandb.config.update(opt.DAMconfig)


        opt.writer = SummaryWriter(log_dir = opt.run_path )
        print(f"[info] tensorboard integration:\ntensorboard --logdir '{opt.data}'\n")
    else:
        
        if opt.wandb:
            wandb.login()
            # wandb.tensorboard.patch(root_logdir=opt.run_path, pytorch=True)
            # sync_tensorboard=True,
            wandb.init( config=opt, project="TEDAM3", entity="hokarami",name=opt.run_name)
            # wandb.config.update(opt.TE_config)
            # wandb.config.update(opt.DAMconfig)


        opt.writer = SummaryWriter(log_dir = opt.run_path )
        print(f"[info] tensorboard integration:\ntensorboard --logdir '{opt.data}'\n")




    opt.label_loss_fun = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(opt.w_pos_label, device=opt.device))

    """ prepare model """
    model = ATHP(
        n_marks=opt.num_marks,
        TE_config = opt.TE_config,
        DAM_config = opt.DAM_config,

        CIF_config = opt.CIF_config,
        next_time_config = opt.next_time_config,
        next_type_config = opt.next_type_config,
        label_config = opt.label_config,

        demo_config = opt.demo_config,

        device=opt.device,

    )
    model.to(opt.device)


    if opt.transfer_learning != '':
        


        files = os.listdir(opt.data+TL_run)

        # find last epoch
        all_epochs=[]
        for file in files:
            if 'model_ep' in file:
                all_epochs.append(int(re.findall(r'\d+', file)[0]))
        if len(all_epochs)==0:
            raise Exception('No models were found for transfer learning!!!')
        MODEL_PATH = opt.data+TL_run+'/model_ep'+str(max(all_epochs))+'.pkl'


        checkpoint = torch.load(MODEL_PATH)

        
        # for para in model.parameters():
        #     para.requires_grad = False

        load_module(model, checkpoint, modules=opt.all_transfered_modules, to_freeze=True)
        print('### [info] all transfered modules: ',opt.all_transfered_modules)
        load_module(model, checkpoint, modules= opt.freezed_modules, to_freeze=False)
        print('### [info] unfreezed modules: ',opt.freezed_modules)



        

       

        # model.A_reg.requires_grad=True
        


    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                        opt.lr,momentum=0.01, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)



    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """




    

    max_obj_val = 0
    if opt.state:

        max_obj_val = train(model, opt.trainloader, opt.validloader, opt.testloader, optimizer, scheduler, opt.pred_loss_func, opt, trial)

    else:

        max_obj_val = train(model, opt.trainloader, opt.validloader, opt.testloader, optimizer, scheduler, opt.pred_loss_func, opt, trial)
    

    if isinstance(trial,optuna.trial._trial.Trial) and trial.should_prune():
        if opt.wandb:
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
        raise optuna.exceptions.TrialPruned()
    
    if opt.wandb:
        
        # report the final validation accuracy to wandb
        wandb.run.summary["max_obj_val"] = max_obj_val
        wandb.run.summary["status"] = "completed"
        wandb.finish(quiet=True)

    return max_obj_val

if __name__ == '__main__':
    

    main()
