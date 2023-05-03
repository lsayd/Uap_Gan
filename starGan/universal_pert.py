import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import attacks




def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v=v.cpu()
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

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

class universal(object):
    def __init__(self, model_G=None,model_D=None, device=None):
        self.model_G = model_G.to(device)
        self.model_D = model_D.to(device)
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

    def universal_perturbation(self, data_loader,selected_attrs, delta=0.1, xi=10, p=np.inf):
        """
        :param xi: controls the l_p magnitude of the perturbation (default = 10)

        :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

        :param delta: controls the desired fooling rate (default = 90% fooling rate)

        :return: the universal perturbation.

        """
        v=torch.tensor(0)

        fooling_rate = 0.0

        itr = 0
        for i,(x_real,c_org) in enumerate(data_loader):
            x_real=x_real.to(self.device)

            c_trg_list=create_labels(
                c_org, 5, 'CelebA', selected_attrs)
              # Translated images.

            with torch.no_grad():
                x_real_mod = x_real
                c_trg=c_trg_list[0].to(self.device)
                gen_noattack, gen_noattack_feats = self.model_G(
                    x_real_mod, c_trg)

            # 获得输出图像
            X=x_real.clone().detach_().cuda()
            X.requires_grad=True
            output, _=self.model_G(X,c_trg)


            # Compute adversarial perturbation

            #PGD
            pgd_attack = attacks.LinfPGDAttack(
                model=self.model_G, device=self.device, feat=None)
            x_adv, dr = pgd_attack.perturb(
                x_real, gen_noattack, c_trg)
            # dr, iter = self.deepfool(x_real + v, gen_noattack, c_trg)

            # Make sure it converged...
            v = v.cpu()
            dr=dr.cpu()
            v = v + dr

            # Project on l_p ball
            v = proj_lp(v, xi, p)


            with torch.no_grad():
                v=v.cuda()
                x_real_mod = x_real+v
                x_real_mod = x_real_mod.to(self.device)
                c_trg=c_trg_list[0].to(self.device)
                gen_attack, gen_attack_feats = self.model_G(
                    x_real_mod, c_trg)
            v=v.cuda()
            # Initialize Metrics
            l1_error, l2_error = 0.0, 0.0
            nums, fooling = 0, 0
            for i, (x_real, c_org) in enumerate(data_loader):
                nums += 1;
                with torch.no_grad():
                    x_real_mod = x_real
                    c_trg = c_trg_list[0].to(self.device)
                    gen_noattack, gen_noattack_feats = self.model_G(
                        x_real_mod, c_trg)
                    v = v.cpu()
                    x_real_mod = x_real + v
                    x_real_mod = x_real_mod.to(self.device)
                    gen_attack, gen_attack_feats = self.model_G(
                        x_real_mod, c_trg)
                l2_error += F.mse_loss(gen_attack, gen_noattack)
                if(l2_error>60):
                    fooling += 1;
                if(nums>100):
                    break
            itr = itr + 1
            print("itr:%d"%(itr))
            print("l2_error:%f" % (l2_error))

            if fooling/nums>1-delta and itr>1000: break
           #  if  itr > 1000: break


        return v


