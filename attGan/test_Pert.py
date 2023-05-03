#-*- coding = utf-8 -*-
#@Time : 2023/3/6 下午4:51
#@Author : lv
#@File : solverUniversal_AttGan.py
#@Software : PyCharm

# """Entry point for testing AttGAN network."""
#
# import argparse
# import json
# import os
# from os.path import join
# import torch
# import torch.utils.data as data
# import torchvision.utils as vutils
#
# from attgan import AttGAN
# from data import check_attribute_conflict
# from helpers import Progressbar
# from utils import find_model
# from solverUniversal_AttGan import SolverUniversal
# import sys
# sys.path.append(r'D:\lkq\UniversalPert_Gan\stargan')
# from data_loader import CelebA,get_loader
#
#
#
# def parse(args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--experiment_name', dest='experiment_name', required=True)
#     parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
#     parser.add_argument('--num_test', dest='num_test', type=int)
#     parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
#     parser.add_argument('--custom_img', action='store_true')
#     parser.add_argument('--custom_data', type=str, default='./data/custom')
#     parser.add_argument('--attr_path', type=str, default='./data/list_attr_celeba.txt')
#     parser.add_argument('--gpu', action='store_true')
#     parser.add_argument('--multi_gpu', action='store_true')
#     parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
#                         default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
#     parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
#     return parser.parse_args(args)
#
# args_ = parse()
# print(args_)
# with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
#     args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
#
#
# args.test_int = args_.test_int
# args.num_test = args_.num_test
# args.gpu = args_.gpu
# args.load_epoch = args_.load_epoch
# args.multi_gpu = args_.multi_gpu
# args.custom_img = args_.custom_img
# args.custom_data = args_.custom_data
# args.attrs_path = args_.attr_path
# args.selected_attrs=args_.selected_attrs
# args.c_dim=args_.c_dim
# args.n_attrs = len(args.attrs)
# args.betas = (args.beta1, args.beta2)
#
# print(args)
#
# # celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
# #                                    config.celeba_crop_size, config.image_size, config.batch_size,
# #                                    'CelebA', config.mode, config.num_workers)
# #     def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
# celeba_loader=get_loader(args.custom_data,args.attr_path,args.selected_attrs,mode='test')
# solver = SolverUniversal(celeba_loader,args)
# solver.test_universal_OnImages()

import argparse
import json
import os
from os.path import join

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    return parser.parse_args(args)


args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_int = args_.test_int
args.num_test = args_.num_test
args.gpu = args_.gpu
args.load_epoch = args_.load_epoch
args.multi_gpu = args_.multi_gpu
args.custom_img = args_.custom_img
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

print(args)

if args.custom_img:
    output_path = join('output', args.experiment_name, 'custom_testing')
    from data import Custom

    print(1111111111)
    print(args.custom_data)
    test_dataset = Custom(args.custom_data, args.custom_attr, args.img_size, args.attrs)
else:
    output_path = join('output', args.experiment_name, 'sample_testing')
    if args.data == 'CelebA':
        from data import CelebA

        test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)

    if args.data == 'CelebA-HQ':
        from data import CelebA_HQ

        test_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'test',
                                 args.attrs)
os.makedirs(output_path, exist_ok=True)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=1, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
if args.num_test is None:
    print('Testing images:', len(test_dataset))
else:
    print('Testing images:', min(len(test_dataset), args.num_test))

attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
progressbar = Progressbar()

attgan.eval()
for idx, (img_a, att_a) in enumerate(test_dataloader):
    if args.num_test is not None and idx == args.num_test:
        break

    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)

    att_b_list = [att_a]
    for i in range(args.n_attrs):
        tmp = att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)

    with torch.no_grad():
        samples = []
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
            samples.append(attgan.G(img_a, att_b_))
        samples = torch.cat(samples, dim=3)
        if args.custom_img:
            out_file = test_dataset.images[idx]
        else:
            out_file = '{:06d}.jpg'.format(idx + 182638)
        vutils.save_image(
            samples, join(output_path, out_file),
            nrow=1, normalize=True, range=(-1., 1.)
        )
        print('{:s} done!'.format(out_file))
