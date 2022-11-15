import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from scipy import misc
import numpy as np

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                m.weight.data.normal_(0.0, 0.02)
                #init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            #init.normal_(m.weight.data, 1.0, gain)
            m.weight.data.normal_(1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = Generator()  
    return init_net(net,init_type, init_gain, gpu_id)

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


class PyramidAttention(nn.Module):
    def __init__(self):
        super(PyramidAttention, self).__init__()
        self.ksize = 3
        self.stride = 1
        self.res_scale = 1
        self.softmax_scale = 10
        self.scale = [1-i/10 for i in range(3)]
        self.average = True
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = BasicBlock(default_conv,512,256, 1, bn=False, act=nn.PReLU())
        self.conv_match = BasicBlock(default_conv,512, 256, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(default_conv,512, 512,1,bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input
        #theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base,1,dim=0)
        # patch size for matching 
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)):    
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
            #feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            #sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            #feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            #sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)
        y = []
        for idx, xi in enumerate(input_groups):
            #group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))],dim=0)  # [L, C, k, k]
            #normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi/ max_wi
            #matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1,wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi*self.softmax_scale, dim=1)          
            if self.average == False:
                yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()           
            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))],dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride,padding=1)/4.
            y.append(yi)     
        y = torch.cat(y, dim=0)+res*self.res_scale  # back to the mini-batch
        return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.endecoder2 = endecoder2()
        
    def forward(self, input):
        x,edge,struct,structedge =self.endecoder2(input)
        return x,edge,struct,structedge

class endecoder2(nn.Module):
    def __init__(self):
        super(endecoder2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv12 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv22 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv24 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512*2, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512*2, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256*2+6, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128*2, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 3, 5, 1, 2),
            nn.Tanh()
            )
        self.up = ConvUp(512, 256, 1, 1)
        self.up256 = ConvUp(256, 128, 1, 1)
        self.up512 = ConvUp(128, 64, 1, 1)
        self.up256edge = ConvUp(256, 128, 1, 1)
        self.up512edge = ConvUp(128, 64, 1, 1)
        self.conva = nn.Sequential(
            nn.Conv2d(768, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(768, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(768, 3, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.fusion1edge = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv_11 = nn.Sequential(
            nn.Conv2d(6, 3, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(256, 256, 7, 1, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(6, 6, 3, 2, 1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(6, 6, 3, 2, 1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.PyramidAttention = PyramidAttention()

    def forward(self,input):
        x = self.conv1(input)#512*512*64
        x11 = self.conv11(x)#256*256*128
        x12 = self.conv12(x11)#128*128*256
        res1 = x#512*512*64
        x = self.conv2(x)#256*256*128
        x22 = self.conv22(x)#128*128*256
        res2 = x#256*256*128
        x = self.conv3(x)#128*128*256
        x32 = x#128*128*256
        res3 = x#128*128*256
        x = self.conv4(x)#64*64*512
        x42 = self.up(x, (128, 128))#128*128*256
        res4 = x#64*64*512
        x = self.conv5(x)#32*32*512
        x52 = self.up(x, (128, 128))#128*128*256
        res5 = x#32*32*512
        x = self.conv6(x)#16*16*512
        x62 = self.up(x, (128, 128))#128*128*256
        edge = torch.cat([x12,x22,x32],1)#128*128*768
        edge = self.conv_1(edge)#128*128*256
        edge3 = self.conv_3(edge)#128*128*256
        edge5 = self.conv_5(edge)#128*128*256
        edge7 = self.conv_7(edge)#128*128*256
        edge = torch.cat([edge3,edge5,edge7],1)#128*128*768
        edge = self.fusion(edge)#128*128*3
        struct = torch.cat([x42,x52,x62],1)#128*128*768
        struct = self.conv_1(struct)#128*128*256
        struct3 = self.conv_3(struct)#128*128*256
        struct5 = self.conv_5(struct)#128*128*256
        struct7 = self.conv_7(struct)#128*128*256
        struct = torch.cat([struct3,struct5,struct7],1)#128*128*768
        struct = self.fusion(struct)#128*128*3
        structedge = torch.cat([struct,edge],1)#128*128*6
        x = self.deconv6(x)#32*32*512
        x = self.PyramidAttention(x)
        #x = x + structedge321
        x = torch.cat([x,res5],1)
        x = self.deconv5(x)
        #x = x + structedge641  
        x = torch.cat([x,res4],1)
        x = self.deconv4(x)
        #x = x + structedge 
        x = torch.cat([x,res3,structedge],1)
        x = self.deconv3(x)
        #x = x + structedge2564
        x = torch.cat([x,res2],1)
        x = self.deconv2(x)
        #x = x + structedge5125
        x = torch.cat([x,res1],1)
        x = self.deconv1(x)
        return x,edge,struct,structedge

class ConvUp(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel,
                              stride, padding, dilation, groups, bias)
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode='bilinear')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

def define_D(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):   
    #net = NLayerDiscriminator()
    net = Discriminator()
    return init_net(net,init_type, init_gain, gpu_id)

class HSResnet(nn.Module):
    def __init__(self):
        super(HSResnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 80, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv21 = nn.Conv2d(80, 16, 1, 1, 0)
        self.conv22 = nn.Conv2d(80, 16, 1, 1, 0)
        self.conv23 = nn.Conv2d(80, 16, 1, 1, 0)
        self.conv24 = nn.Conv2d(80, 16, 1, 1, 0)
        self.conv25 = nn.Conv2d(80, 16, 1, 1, 0)
        self.conv31 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv32 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv33 = nn.Sequential(
            nn.Conv2d(28, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv34 = nn.Sequential(
            nn.Conv2d(30, 30, 3, 1, 1),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv41 = nn.Conv2d(16, 8, 1, 1, 0)
        self.conv42 = nn.Conv2d(16, 8, 1, 1, 0)
        self.conv43 = nn.Conv2d(24, 12, 1, 1, 0)
        self.conv44 = nn.Conv2d(24, 12, 1, 1, 0)
        self.conv45 = nn.Conv2d(28, 14, 1, 1, 0)
        self.conv46 = nn.Conv2d(28, 14, 1, 1, 0)
        self.conv6 = nn.Sequential(
            nn.Conv2d(80, 64, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
            )
    def forward(self, x):
        res = x
        x1 = self.conv1(x)
        x21 = self.conv21(x1)
        x22 = self.conv22(x1)
        x31 = self.conv31(x22)
        x41 = self.conv41(x31)
        x42 = self.conv42(x31)
        x23 = self.conv23(x1)
        x51 = torch.cat([x42,x23],1)
        x32 = self.conv32(x51)
        x43 = self.conv43(x32)
        x44 = self.conv44(x32)
        x24 = self.conv24(x1)
        x52 = torch.cat([x44,x24],1)
        x33 = self.conv33(x52)
        x45 = self.conv45(x33)
        x46 = self.conv46(x33)
        x25 = self.conv25(x1)
        x53 = torch.cat([x46,x25],1)
        x34 = self.conv34(x53)
        x54 = torch.cat([x21,x41,x43,x45,x34],1)
        x54 = self.conv6(x54)
        x54 = x54 + res
        return x54

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 1, 4, 1, 1),
            nn.Sigmoid()
            )
        self.HSResnet = HSResnet()
        
    def forward(self, x):
        n_layers = 3
        x = self.conv1(x)
        for i in range(n_layers):
            x = self.HSResnet(x)
            x = self.conv2(x)
        x = self.conv3(x)
        return x

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



