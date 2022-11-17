import os
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import pickle

import torch
from utils.lr_layer_scale import param_groups_lrd
from .misc import NativeScalerWithGradNormCount as NativeScaler
## for optimizaer
import random

from torch import optim as optim

def fix_seed(seed):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name))

    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)



def inner_valid(args, model, test_loader):
    eval_losses = AverageMeter()

    print("++++++ Running Validation of client", args.single_client, "++++++")
    model.eval()
    all_preds, all_label = [], []

    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(x)
                eval_loss = loss_fct(logits, y)

            eval_losses.update(eval_loss.item())

            if args.num_classes > 1:
                preds = torch.argmax(logits, dim=-1)
            else:
                preds = logits

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
    all_preds, all_label = all_preds[0], all_label[0]
    if not args.num_classes == 1:
        eval_result = simple_accuracy(all_preds, all_label)
    else:
        # eval_result =  mean_absolute_error(all_preds, all_label)
        eval_result =  mean_squared_error(all_preds, all_label)

    model.train()

    return eval_result, eval_losses

def metric_evaluation(args, eval_result):
    if args.num_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag

def valid(args, model, val_loader,  test_loader = None, TestFlag = False):
    # Validation!
    eval_result, eval_losses = inner_valid(args, model, val_loader)

    print("Valid Loss: %2.5f" % eval_losses.avg, "Valid metric: %2.5f" % eval_result)
    
    if metric_evaluation(args, eval_result):
        if args.save_model_flag:
            save_model(args, model)

        args.best_acc[args.single_client] = eval_result
        args.best_eval_loss[args.single_client] = eval_losses.val
        print("The updated best metric of client", args.single_client, args.best_acc[args.single_client])

        if TestFlag:
            test_result, eval_losses = inner_valid(args, model, test_loader)
            args.current_test_acc[args.single_client] = test_result
            print('We also update the test acc of client', args.single_client, 'as',
                    args.current_test_acc[args.single_client])
    else:
        print("Donot replace previous best metric of client", args.best_acc[args.single_client])

    args.current_acc[args.single_client] = eval_result


def optimization_fun(args, model, lr=None):
    if not lr:  #default
        lr = args.lr

    if 'vit' in args.net_name and args.optimizer_type == 'adamw':   #for all ViTs have layer decay
        assert args.layer_decay, 'no layer decay args'
        
        param_groups = param_groups_lrd(model, args.weight_decay,
                    no_weight_decay_list=model.no_weight_decay(),
                    layer_decay=args.layer_decay
                    )
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
        print('----caution: use weight decay----')
        return optimizer

    # Prepare optimizer, scheduler
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=lr, weight_decay=args.weight_decay)
    else:
        assert False, 'please choose optimizer between sgd and adamw'
    
    return optimizer


def Partial_Client_Selection(args, model):

    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    args.learning_rate_record = {}
    args.t_total = {}
    loss_scaler_all = {}

    build_option = args.cur_selected_clients if args.dataset=='inat' else args.dis_cvs_files

    for proxy_single_client in build_option:   #just build useful model
        #print(dict(model.state_dict()))
        model_all[proxy_single_client] = pickle.loads(pickle.dumps(model)).cpu()
        optimizer_all[proxy_single_client] = optimization_fun(args, model_all[proxy_single_client])
        
        if args.drop_last:
            tmp_rounds = math.floor(args.clients_with_len[proxy_single_client]/args.batch_size) 
        else:
            tmp_rounds = math.ceil(args.clients_with_len[proxy_single_client]/args.batch_size) 
        #iteration
        args.t_total[proxy_single_client] = tmp_rounds* args.max_communication_rounds* args.E_epoch

        loss_scaler_all[proxy_single_client] = NativeScaler()

        args.learning_rate_record[proxy_single_client] = []

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.dis_cvs_files}

    return model_all, optimizer_all, loss_scaler_all


def average_model(args,  model_avg, model_all):
    model_avg.cpu()
    print('Calculate the model avg----')
    if args.update_momentum:
        param_ori = deepcopy(dict(model_avg.state_dict()))
    params = dict(model_avg.state_dict())

    # for name, value in model_state_dict.items():
    for name, param in params.items():
        for client in range(len(args.cur_selected_clients)):  #aggregate selected
            single_client = args.cur_selected_clients[client]

            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()

            if client == 0:
                tmp_param_data = dict(model_all[single_client].state_dict())[
                                     name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(model_all[single_client].state_dict())[
                                     name].data * single_client_weight
        if args.update_momentum:
            params[name].data.copy_((1-args.update_momentum)* param_ori[name].data + args.update_momentum *tmp_param_data)
        else:
            params[name].data.copy_(tmp_param_data)
    
    print('Update each client model parameters----')
    
    for single_client in args.cur_selected_clients:  #init selected models
        tmp_params = dict(model_all[single_client].state_dict())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)    
    
    print('Complete update')
    model_avg.to(args.device)

def init_model(args, model_avg, model_all):
    model_avg.cpu()
    params = dict(model_avg.state_dict())
    print('Update each client model parameters----')

    for single_client in args.cur_selected_clients:  #init selected models
        tmp_params = dict(model_all[single_client].state_dict())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)