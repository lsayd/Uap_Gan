#-*- coding = utf-8 -*-
#@Time : 2023/3/8 下午8：16
#@Author : lv
#@File : solverUniversal_AttGan.py
#@Software : PyCharm
from os.path import join

from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import attacks

from PIL import ImageFilter
from PIL import Image
from torchvision import transforms

import sys
from utils import find_model

sys.path.append(r'D:\lkq\UniversalPert_Gan\stargan')
from data_loader import CelebA,get_loader
from universal_pert import universal
from attgan import AttGAN


def create_labels( c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                # Reverse attribute value.
                c_trg[:, i] = (c_trg[:, i] == 0)


        c_trg_list.append(c_trg)
    return c_trg_list

class SolverUniversal(object):
    def __init__(self,celeba_loader,args):
        self.celeba_loader = celeba_loader
        self.args=args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.G=None

    def restore_model(self):
        """Restore the trained generator and discriminator."""
        attgan = AttGAN(self.args)
        attgan.load(find_model(join('output', self.args.experiment_name, 'checkpoint'), self.args.load_epoch))
        self.G=attgan.G
        print(111111111111111)


    def get_universal_perturbation(self):
        self.restore_model()
        file_perturbation = os.path.join('data', 'universal.npy')

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader


        model_G = self.G
        universal_pert = universal(model_G=model_G, device=self.device)

        v = universal_pert.universal_perturbation(data_loader, self.selected_attrs)

        np.save(os.path.join(file_perturbation), v)
        print("saved successfully")

    def test_universal_OnImages(self):

        print('>> Testing the universal perturbation on  images')
        self.restore_model()
        # Set data loader.
        data_loader = self.celeba_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):

            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            x_arr = np.array(x_real.cpu())
            print(x_arr)
            c_trg_list = create_labels(
                c_org, 5 , selected_attrs=self.args.selected_attrs)

            # Translated images.
            x_fake_list = [x_real]

            for idx, c_trg in enumerate(c_trg_list):
                print('image', i, 'class', idx)
                with torch.no_grad():
                    x_real_mod = x_real
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(
                        x_real_mod, c_trg)

                # Attacks
                #    x_adv, perturb = universal.perturb(
                #        x_real, self.D.forward(), self.grad_fs(),c_trg)
                #
                #    x_adv = x_real + perturb
                file_perturbation = os.path.join('data', 'universal.npy')
                v = np.load(file_perturbation)
                v = torch.tensor(v)
                # x_arr=np.array(x_real.cpu())
                # print(x_arr)
                x_adv = x_real + v.cuda()
                # print(x_adv)
                # x_adv=Image.fromarray(x_adv)
                # x_adv=x_adv.cuda()
                with torch.no_grad():
                    gen, _ = self.G(x_adv, c_trg)

                    # Add to lists
                    # x_fake_list.append(blurred_image)
                    x_fake_list.append(x_adv)
                    # x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            save_dir = r"\D:\lkq\UniversalPert_Gan\AttGAN\results\universal_starGan"
            result_path = os.path.join(
                save_dir, '{}-images.jpg'.format(i + 1))
            save_image(self.denorm(x_concat.data.cpu()),
                       result_path, nrow=1, padding=0)
            if i == 49:  # stop after this many images
                break

        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
                                                                                                             l1_error / n_samples,
                                                                                                             l2_error / n_samples,
                                                                                                             float(
                                                                                                                 n_dist) / n_samples,
                                                                                                             l0_error / n_samples,
                                                                                                             min_dist / n_samples))
