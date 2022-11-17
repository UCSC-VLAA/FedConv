# coding=utf-8
import os
import argparse
import numpy as np
from copy import deepcopy
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pickle

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.data_utils import DatasetFLViT, create_dataset_and_evalmetrix
from utils.util import Partial_Client_Selection, valid, average_model, fix_seed
from utils.start_config import initization_configure, _worker_init
from utils.scheduler import adjust_learning_rate

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    args.drop_last = True 

    # Prepare dataset
    create_dataset_and_evalmetrix(args)

    if args.split_type != 'real_test':
        testset = DatasetFLViT(args, phase = 'test' )
        test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.batch_size, num_workers=args.num_workers)

    # get the union val dataset,
    if args.dataset == 'cifar10':
        valset = DatasetFLViT(args, phase = 'val' )
        val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=args.num_workers)

    model_all, optimizer_all, loss_scaler_all = Partial_Client_Selection(args, model)
    model_avg = pickle.loads(pickle.dumps(model))

    print("=============== Begin training ===============")

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        loss_fct = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        loss_fct = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(loss_fct))

    tot_clients = args.dis_cvs_files
    epoch = -1

    while True:
        epoch += 1
        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            args.cur_selected_clients = args.dis_cvs_files
        else:
            args.cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()

        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in args.cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]
        
        for cur_single_client in args.cur_selected_clients:

            args.single_client = cur_single_client
            args.clients_weightes[cur_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens
            
            trainset = DatasetFLViT(args, phase='train')
            train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.batch_size, 
                                      num_workers=args.num_workers, drop_last=args.drop_last,pin_memory=True,
                                      worker_init_fn=_worker_init, 
                                      )

            model = model_all[cur_single_client]
            model = model.to(args.device).train()

            optimizer = optimizer_all[cur_single_client]
            loss_scaler = loss_scaler_all[cur_single_client]

            print('Train the client', cur_single_client, 'of communication round', epoch)

            for inner_epoch in range(args.E_epoch):
                for step, batch in enumerate(train_loader):  # batch = tuple(t.to(args.device) for t in batch)
                    args.global_step_per_client[cur_single_client] += 1
                    adjust_learning_rate(optimizer, step / len(train_loader) + epoch, args)

                    batch = tuple(t.to(args.device) for t in batch)
                    
                    x, y = batch
                    if mixup_fn is not None:
                        if x.size(0) % 2 == 1:  #odd batch, drop last one
                            x = x[:-1]
                            y = y[:-1]
                        x, y = mixup_fn(x, y)

                    with torch.cuda.amp.autocast():
                        predict = model(x)
                        loss = loss_fct(predict.view(-1, args.num_classes), y)
                        
                        if args.mu:  #Prox
                            proximal_term = 0.0
                            for param_cur, param_avg in zip(model.parameters(), model_avg.parameters()):
                                proximal_term += (param_cur - param_avg).norm(2)
                            #loss_2 = loss_2.to(args.device)
                            loss = loss + (args.mu / 2) * proximal_term
                            
                        loss = loss/args.update_freq
                    
                    loss_num = loss.item()
                    
                    clip = args.max_grad_norm if args.clip_grad else None
                    agc = args.agc
                    
                    if((step+1)%args.update_freq)==0:
                        loss_scaler(loss, optimizer, clip_grad=clip, agc=agc,
                            parameters=model.parameters(), create_graph=False,
                            update_grad=True)
                        optimizer.zero_grad()
                    else:
                        loss_scaler(loss, optimizer, clip_grad=clip, agc=agc,
                            parameters=model.parameters(), create_graph=False,
                            update_grad=False)
                    
                    torch.cuda.synchronize()
                    
                    writer.add_scalar(cur_single_client + '/lr', scalar_value=optimizer.param_groups[0]['lr'],
                                      global_step=args.global_step_per_client[cur_single_client])
                    writer.add_scalar(cur_single_client + '/loss', scalar_value=loss_num,
                                      global_step=args.global_step_per_client[cur_single_client])


                    args.learning_rate_record[cur_single_client].append(optimizer.param_groups[0]['lr'])

                    if (step+1 ) % args.output_freq == 0:
                        print(cur_single_client, step,':', len(train_loader),'inner epoch', inner_epoch, 'round', epoch,':',
                              args.max_communication_rounds, 'loss', loss_num, 'lr', optimizer.param_groups[0]['lr'])
                    
            model.eval()
        
        # average model
        average_model(args, model_avg, model_all)

        np.save(args.output_dir + '/learning_rate.npy', args.learning_rate_record)
        
        # then evaluate, eval all
        for cur_single_client in args.dis_cvs_files:
            args.single_client = cur_single_client
            model = model_all[cur_single_client]
            
            model.to(args.device)

            if args.dataset == 'COVIDfl' and args.split_type == 'real_test':
                valset = DatasetFLViT(args, phase='test')
                val_loader_proxy_clients = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size,
                                        num_workers=args.num_workers)
                valid(args, model, val_loader_proxy_clients, test_loader=None, TestFlag=False)

            elif args.dataset == 'COVIDfl':   #union validation
                valid(args, model, val_loader=test_loader, test_loader=None, TestFlag=False)

            else:  # for Cifar10 dataset
                val_loader_proxy_clients = val_loader
                valid(args, model, val_loader_proxy_clients, test_loader, TestFlag=True)

            #model.cpu() 

        tmp_round_acc = [val for val in args.current_acc.values() if not val == []]

        tmp_weight_round_acc = []
        for client in args.dis_cvs_files:
            tmp_weight_round_acc.append(args.current_acc[client]*args.clients_weightes[client])
        print('weight_val_acc',np.asarray(tmp_weight_round_acc).sum())

        tmp_round_test_acc = [test for test in args.current_test_acc.values() if not test == []]

        mean_val = np.asarray(tmp_round_acc).mean()
        weight_val = np.asarray(tmp_weight_round_acc).sum()
        val_record = deepcopy(args.current_acc)
        val_record['weighted_acc'] = weight_val
        val_record['mean_acc'] = mean_val

        writer.add_scalar("val/average_accuracy", scalar_value=mean_val, global_step=epoch)
        writer.add_scalar("val/weight_average_accuracy", scalar_value=weight_val, global_step=epoch)
        writer.add_scalar("test/average_accuracy", scalar_value=np.asarray(tmp_round_test_acc).mean(), global_step=epoch)
        
        args.record_val_acc = args.record_val_acc.append(val_record, ignore_index=True)
        args.record_val_acc.to_csv(os.path.join(args.output_dir, 'val_acc.csv'))
        args.record_test_acc = args.record_test_acc.append(args.current_test_acc, ignore_index=True)
        args.record_test_acc.to_csv(os.path.join(args.output_dir, 'test_acc.csv'))

        #if args.global_step_per_client[proxy_single_client] >= args.t_total[proxy_single_client]:  #last
        if epoch >= args.max_communication_rounds-1 :
            break

    writer.close()
    print("================End training! ================ ")


def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type = str, default="mconv",  help="Basic Name of this run with detailed network-architecture selection. ")
    parser.add_argument("--dataset", choices=["cifar10", 'COVIDfl'], default="cifar10", help="Which dataset.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Where is dataset located.")

    parser.add_argument('--Pretrained', action='store_true', help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="", help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]")
    parser.add_argument('--use_ema', action='store_true', default=False, help='whether use model ema')
    

    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--clip_grad', action='store_true', default=False, help="whether gradient clip or not")
    parser.add_argument("--max_grad_norm", default=10., type=float,  help="Max gradient norm.")
    parser.add_argument('--agc', default=None, type=float, help="The value of adaptive grad clip")

    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float, help='layer scale value')
    parser.add_argument('--update_freq', default=1, type=int, help='optimizer step frequency')
    

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=32, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization") #99999

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Epoches of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--layer_scale", default=1., type=int, help="Number of layer scale")
    parser.add_argument('--layer_decay', type=float, default=None, metavar='PCT', help='Value of layer decay for ViT')
    parser.add_argument("--lr", default=3e-2, type=float,  help="The initial learning rate")
    parser.add_argument("--min_lr", default=1e-6, type=float,  help="lower lr bound for cyclic schedulers that hit 0")


    ## FL related parameters
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,  help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=-1, type=int, help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3",'real_test'], default="split_3", help="Which data partitions to use")


    # Augmentation & regularization parameters
    #Aug
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    #stochastic depth
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    parser.add_argument('--mu', default=None, type=float, help='mu for fedProx')
    parser.add_argument('--share', action='store_true', default=False, help='whether turn on share')
    parser.add_argument('--update_momentum', default=None, type=float, help='momentum for FedAvgM')
    #save 
    parser.add_argument('--output_freq', type=int, default=10, help='')
    parser.add_argument("--save_model_flag",  action='store_true', default=False,  help="Save the best model for each client.")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--msg", default="", type=str, help="The output directory with message.")
    
    args = parser.parse_args()
    
    fix_seed(args.seed)

    # Initialization
    model = initization_configure(args)

    # Training, Validating, and Testing
    train(args, model)


    message = '\n \n ==============Start showing final performance ================= \n'
    message += 'Final union val accuracy is: %2.5f  \n' %  \
                   (np.asarray(list(args.current_acc.values())).mean())
    message += 'Final union test accuracy is: %2.5f  \n' %  \
                   (np.asarray(list(args.current_test_acc.values())).mean())
    message += "================ End ================ \n"


    with open(args.file_name, 'a+') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)



if __name__ == "__main__":
    main()
