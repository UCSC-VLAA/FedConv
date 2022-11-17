import os
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from timm.models import create_model
from models import build_model
    
def _worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    np.random.seed(worker_info.seed % (2**32-1))

def print_options(args, model):
    model.eval()
    
    #print(model)
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    inputs = torch.rand(1,3,224,224).to(args.device)
    #flops, _ = profile(model, inputs=(inputs, ))

    flops = FlopCountAnalysis(model, inputs).total()

    message += "================ FL train of %s with total model parameters: %3.2fM (FLOPs: %3.2fG)================\n" % (args.net_name, num_params, flops/1000000000)
    message += "pid:{} \n".format(os.getpid())
    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '
    
    ## save to disk of current log

    args.file_name = os.path.join(args.output_dir, 'log_file.txt')

    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)
    
    #assert False
    
    model.train()

def initization_configure(args):
    # args.device = torch.device("cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = 0) if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'COVIDfl':
        args.num_classes = 3
    elif args.dataset == 'inat':
        args.num_classes = 1010
    else:
        assert False, 'please choose right number of classes'


    # Set model type related parameters
    if 'fedconv' in args.net_name:   #all self pretrain model
        model = create_model(args.net_name,
                            pretrained=False,
                            drop_path_rate=args.drop_path,
                            layer_scale_init_value=args.layer_scale_init_value)

        if args.Pretrained:
            assert args.pretrained_dir, 'No pretrain dir!'
            checkpoint = torch.load(args.pretrained_dir, map_location='cpu')
            if args.use_ema:
                checkpoint = checkpoint['model_ema']
            else:
                checkpoint = checkpoint['model']
                model.load_state_dict(checkpoint)

        model.head.fc = nn.Linear(model.head.fc.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    elif 'convnext' in args.net_name:  #for official pretrain
        print(f'We use {args.net_name}')
        model = create_model(args.net_name,
                            pretrained=args.Pretrained,
                            num_classes=args.num_classes,
                            drop_path_rate=args.drop_path,
                            ls_init_value=args.layer_scale_init_value
                            )
        model.to(args.device)

    else:
        print(f'We use {args.net_name}')
        model = create_model(args.net_name,
                            pretrained=args.Pretrained,
                            num_classes=args.num_classes,
                            drop_path_rate=args.drop_path,
                            )
        model.to(args.device)

    # set output parameters
    print(args.optimizer_type)        
    args.name = args.net_name + '_' + args.msg + '_' + args.split_type + '_lr_' + str(args.lr) + '_Pretrained_' \
            + str(args.Pretrained) + "_optimizer_" + str(args.optimizer_type) + '_WUPE_'  + str(args.warmup_epochs) \
            + '_Round_' + str(args.max_communication_rounds) + '_Eepochs_' + str(args.E_epoch) + '_Seed_' + str(args.seed)


    args.output_dir = os.path.join('output', args.dataset, args.name)
    os.makedirs(args.output_dir, exist_ok=True)

    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}

    return model


