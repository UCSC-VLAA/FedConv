import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
import torch
from torchvision import transforms

import torch.utils.data as data

Image.LOAD_TRUNCATED_IMAGES = True

CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


class DatasetFLViT(data.Dataset):
    def __init__(self, args, phase ):
        super(DatasetFLViT, self).__init__()
        self.phase = phase

        if self.phase == 'train' and args.dataset == 'COVIDfl':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.2)),
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(
                    mean=torch.tensor((0.485, 0.456, 0.406)),
                    std=torch.tensor((0.229, 0.224, 0.225)))
                ])
        elif self.phase == 'train':  #cifar and inat train
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

        elif args.dataset == 'COVIDfl': #covid val
            self.transform = transforms.Compose([
                transforms.Resize([args.img_size, args.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor((0.485, 0.456, 0.406)),
                    std=torch.tensor((0.229, 0.224, 0.225)))
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])



        if args.dataset == "cifar10":
            data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle=True)
            data_all = data_all.item()


            self.data_all = data_all[args.split_type]

            if self.phase == 'train':
                if args.dataset == 'cifar10':
                    self.data = self.data_all['data'][args.single_client]
                    self.labels = self.data_all['target'][args.single_client]
                else:  #celeba
                    self.data = self.data_all['train'][args.single_client]['x']
                    self.labels = data_all['labels']

            else:   #val and test
                if args.dataset == 'cifar10':

                    self.data = data_all['union_' + phase]['data']
                    self.labels = data_all['union_' + phase]['target']

                else:  #celeba
                    if args.split_type == 'real' and phase == 'val':
                        self.data = self.data_all['val'][args.single_client]['x']
                    elif args.split_type == 'central' or phase == 'test':
                        self.data = list(data_all['central']['val'].keys())

                    self.labels = data_all['labels']


        elif args.dataset == 'COVIDfl':
            if args.split_type == 'central':
                cur_clint_path = os.path.join(args.data_path, f'{self.phase}.csv')

            elif self.phase == 'train':
                train_folder = args.split_type.split('_')[0] if not args.share else 'fedshare'
                cur_clint_path = os.path.join(args.data_path, '12_clients', 
                                              train_folder, args.single_client)
            elif args.split_type =='real_test' : 
                cur_clint_path = os.path.join(args.data_path, '12_clients', 
                                              args.split_type, args.single_client)
            else: #union test
                cur_clint_path = os.path.join(args.data_path, f'{self.phase}.csv')

            self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
            self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                          open(os.path.join(args.data_path, 'labels.csv'))}

        elif args.dataset == 'inat':
            if self.phase != 'train':  #only val here
                val_path = os.path.join(args.data_path, 'client_data_mapping',  'val.csv')
                cur_clint = pd.read_csv(val_path)
            
            elif args.split_type == 'central':
                cur_clint = pd.read_csv(os.path.join(args.data_path, 'client_data_mapping', 'train.csv'))

            else:  #train
                f = pd.read_csv(os.path.join(args.data_path, 'client_data_mapping', 'train.csv'))
                cur_clint = f[f['client_id'] == args.single_client]

            self.img_paths = []
            self.labels = []
            for index in cur_clint.index:
                self.img_paths.append(cur_clint.loc[index]['data_path'])
                self.labels.append(cur_clint.loc[index]['label_id'])

        self.args = args


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.args.dataset == 'cifar10':
            img, target = self.data[index], self.labels[index]
            img = Image.fromarray(img)
            
        elif self.args.dataset == 'COVIDfl':
            index = index % len(self.img_paths)
            path = os.path.join(self.args.data_path, self.phase, self.img_paths[index])
            name = self.img_paths[index]

            target = self.labels[name]
            target = np.asarray(target).astype('int64')

            img = np.array(Image.open(path).convert("RGB"))
            if img.ndim < 3:
                img = np.stack((img,)*3, axis=-1)
            elif img.shape[2] >= 3:
                img = img[:,:,:3]

            img = Image.fromarray(np.uint8(img))

        elif self.args.dataset == 'inat':
            name = self.img_paths[index]
            target = self.labels[index]
            target = np.asarray(target).astype('int64')

            path = os.path.join(self.args.data_path, name)
            img = Image.open(path).convert("RGB")
            
        if self.transform is not None:
            img = self.transform(img)

        return img,  target


    def __len__(self):
        if self.args.dataset == 'cifar10' or self.args.dataset == 'CelebA':
            return len(self.data)
        else:
            return len(self.img_paths)

def create_dataset_and_evalmetrix(args):

    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = ['central']

    if args.dataset == 'cifar10':
        # get the client with number
        data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle=True)
        data_all = data_all.item()

        data_all = data_all[args.split_type]
        args.dis_cvs_files = [key for key in data_all['data'].keys() if 'train' in key]
        args.clients_with_len = {name: data_all['data'][name].shape[0] for name in args.dis_cvs_files}

    elif args.dataset == 'COVIDfl':
        if args.split_type == 'central':
            args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
        else:
            args.dis_cvs_files = os.listdir(os.path.join(args.data_path, '12_clients', args.split_type))
        
        args.clients_with_len = {}
        
        for single_client in args.dis_cvs_files:
            if args.split_type == 'central':
                img_paths = list({line.strip().split(',')[0] for line in
                              open(os.path.join(args.data_path, args.split_type, single_client))})
            else:
                img_paths = list({line.strip().split(',')[0] for line in
                                  open(os.path.join(args.data_path, '12_clients',
                                                    args.split_type.split('_')[0], single_client))})
            args.clients_with_len[single_client] = len(img_paths)
    
    elif args.dataset == 'inat': 
        args.clients_with_len = {}
        
        if args.split_type == 'central':
            ori_data = pd.read_csv(os.path.join(args.data_path,'client_data_mapping/train.csv'))
            args.clients_with_len['central'] = len(ori_data)

        else:
            train_info = pd.read_csv(os.path.join(args.data_path,'client_data_mapping/train.csv')) 
            args.dis_cvs_files = list(set(train_info['client_id']))

            for single_client in args.dis_cvs_files:
                args.clients_with_len[single_client] = len(train_info[train_info['client_id'] == single_client])

    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    col_name = ['weighted_acc','mean_acc']
    col_name.extend(args.dis_cvs_files)
    args.record_val_acc = pd.DataFrame(columns=col_name)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)

    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}

    for single_client in args.dis_cvs_files:
        args.best_acc[single_client] = 0 if args.num_classes > 1 else 999
        args.current_acc[single_client] = []
        args.current_test_acc[single_client] = []
        args.best_eval_loss[single_client] = 9999
    
    if args.dataset == 'inat':
        args.best_acc['union'] = 0
        args.record_val_acc_oi = pd.DataFrame(columns=['union'])
        args.record_test_acc_oi = pd.DataFrame(columns=['union'])

    if args.num_local_clients == -1:
        args.num_local_clients =  len(args.dis_cvs_files)




