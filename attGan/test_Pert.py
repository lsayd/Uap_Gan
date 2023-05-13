#-*- coding = utf-8 -*-
#@Time : 2023/3/6 下午4:51
#@Author : lv
#@File : solverUniversal_AttGan.py
#@Software : PyCharm


import argparse
import json
import os
from os.path import join

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.utils import save_image
import numpy as np

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model




def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', default='256_shortcut1_inject0_none_hq')
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='D:\lkq\Image')
    parser.add_argument('--custom_attr', type=str, default='D:/lkq/UniversalPert_Gan/AttGAN/data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    return parser.parse_args(args)


args_ = parse()
print(args_)

with open(r'D:\lkq\UniversalPert_Gan\AttGAN\output\256_shortcut1_inject0_none_hq\setting.txt', 'r') as f:
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
    # output_path = join('output', args.experiment_name, 'custom_testing')
    output_path='D:/lkq/UniversalPert_Gan/AttGAN/output/256_shortcut1_inject0_none_hq/custom_testing'
    from data import Custom
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
attgan.load(find_model(join('D:/lkq/UniversalPert_Gan/AttGAN/output', args.experiment_name, 'checkpoint'), 199))
# attgan.load('D:/lkq/UniversalPert_Gan/AttGAN/output/256_shortcut1_inject0_none_hq/checkpoint')
progressbar = Progressbar()

attgan.eval()


def denorm( x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

# Initialize Metrics
l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
n_dist, n_samples = 0, 0
itr=0

for idx, (img_a, att_a) in enumerate(test_dataloader):
    if args.num_test is not None and idx == args.num_test:
        break

    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)

    image_list=[img_a]
    att_b_list = [att_a]
    for i in range(args.n_attrs):
        tmp = att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)

    with torch.no_grad():
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
            v = np.load(r'D:\lkq\disrupting-deepfakes-master\stargan\data\universal1.npy')
            v.resize([1,3,256,256])
            v = torch.from_numpy(v)
            img_a_adv=img_a+0.3*v
            image_list.append(img_a_adv)
            output=attgan.G(img_a,att_b_)
            output_adv=attgan.G(img_a_adv,att_b_)
            image_list.append(output_adv)

            l1_error += F.l1_loss(output, output_adv)
            l2_error += F.mse_loss(output, output_adv)
            l0_error += (output - output_adv).norm(0)
            min_dist += (output -output_adv).norm(float('-inf'))
            if F.mse_loss(output, output_adv) > 0.05:
                n_dist += 1
            n_samples += 1
            # print(l2_error)

    # Save the translated images.
    x_concat = torch.cat(image_list, dim=3)
    save_dir=r"D:\lkq\UapGan_result"
    result_path = os.path.join(
        save_dir, '{}-images.jpg'.format(itr+1))
    save_image(denorm(x_concat.data.cpu()),
               result_path, nrow=1, padding=0)
    itr+=1
    print(itr)# testGit
    if i == 49:     # stop after this many images
        break

# Print metrics
print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
                                                                                                     l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))


