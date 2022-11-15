from __future__ import print_function
import argparse
import os
from math import log10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from networks import define_G,define_D,GANLoss,get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
import torchvision
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--num_D', type=int, default=1, help='number of discriminators to use')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr1', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--lr2', type=float, default=0.0003, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb1', type=int, default=25, help='weight on L1 term in objective')
parser.add_argument('--lamb2', type=int, default=0.5, help='weight on VGG term in objective')
#parser.add_argument('--lamb3', type=int, default=0.1, help='weight on ATT term in objective')
parser.add_argument('--lamb4', type=int, default=10, help='weight on tv term in objective')
#parser.add_argument('--lamb5', type=int, default=5, help='weight on tv term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset, opt.direction)
test_set = get_test_set(root_path + opt.dataset, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G('normal', '0.02', gpu_id=device)
net_d = define_D('normal','0.02',gpu_id=device)
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)


# setup optimizer
#optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_g = optim.Adam(filter(lambda p: p.requires_grad,net_g.parameters()), lr=opt.lr2, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b, real_c, real_d = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        real_c128 = F.interpolate(real_c,scale_factor = 0.25)
        real_d128 = F.interpolate(real_d,scale_factor = 0.25)
        x,edge,struct,structedge = net_g(real_a)
        fake_b=x
        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab.detach())
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5 
        loss_d.backward(retain_graph=True)
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb4
        loss_g_edge = criterionL1(edge, real_c128)#repair
        loss_g_struct = criterionL1(struct, real_d128)#repair
        #structure repair 
        loss_g = loss_g_gan + loss_g_l1 + loss_g_edge + loss_g_struct
        loss_g.backward()

        optimizer_g.step()
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G_l1: {:.4f} Loss_G_edge: {:.4f} Loss_G_gan: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g_l1.item(),loss_g_edge.item(),loss_g_gan.item()))
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)


    #checkpoint
    if epoch % 10 == 0:
        if not os.path.exists("checkpointtwoende"):
            os.mkdir("checkpointtwoende")
        if not os.path.exists(os.path.join("checkpointtwoende", opt.dataset)):
            os.mkdir(os.path.join("checkpointtwoende", opt.dataset))
        net_g_model_out_path = "checkpointtwoende/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpointtwoende/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpointtwoende" + opt.dataset))
