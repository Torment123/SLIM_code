import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x



class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise
    





    

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class InceptionBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 sz_out,
                 groups=4,
                 reduction=16,
                 spatial_dilation=2,
                 with_noise_injection=False):
        super().__init__()
        
        assert ch_in % groups == 0, \
            f'input channels {ch_in} not divisible by {groups}'
        
        
        self.sz_out = sz_out
        self.groups = groups
        
        c_group = ch_in // groups
        c_reduction = c_group // reduction
        
        dilations = [spatial_dilation for i in range(1, groups+1)]
        
        steps = [2] * groups
        
        self.channel_attn = ModuleList([])
        self.spatial_attn = ModuleList([])
        
        for i in range(groups):
            self.channel_attn.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    nn.Linear(c_group,
                           c_reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(c_reduction,
                           ch_out)))
            
            ops = [nn.Conv2d(c_group, c_reduction, kernel_size=1),
                   nn.BatchNorm2d(c_reduction),
                   nn.ReLU(inplace=True)
                  ]
            
            for _ in range(steps[i]):
                ops.extend([
                    nn.Conv2d(c_reduction,
                           c_reduction,
                           kernel_size=3,
                           padding=spatial_dilation,
                           dilation=spatial_dilation),
                    nn.BatchNorm2d(c_reduction),
                    nn.ReLU(inplace=True)
                ])
             
            
            ops.extend([
                nn.Conv2d(c_reduction, 1, kernel_size=1),
                nn.Sigmoid(),
                Interpolate(size=(sz_out[2], sz_out[3]), mode='nearest')
            ])
            
            self.spatial_attn.append(
                nn.Sequential(*ops)
            )
       
    
    def forward(self, x_small, x_big):
        
        
        
        ch_outs = []
        sp_outs = []
        
        xs = torch.split(x_small, x_small.size(1)//self.groups, dim=1)
        
        for x, ch_attn, sp_attn in zip(xs, self.channel_attn, self.spatial_attn):
            ch_outs.append(ch_attn(x).unsqueeze(1))
            sp_outs.append(sp_attn(x))
        
        ch_outs = torch.cat(ch_outs, dim=1).unsqueeze(-1).unsqueeze(-1)
        sp_outs = torch.cat(sp_outs, dim=1).unsqueeze(2)
        
        
        
        style = torch.sum(sp_outs * ch_outs, dim = 1, keepdim=False)
        style = style / (torch.sum(sp_outs, dim=1, keepdim=False) + 1e-6) + 1
        
        return x_big * style
    
    

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt, reals_shape):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.opt = opt
        self.scale_num = opt.scale_num
        
        self.reals_shape = reals_shape
        
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        
        target_shape = reals_shape[self.scale_num]
        
        target_shape = list(target_shape)
        
        target_shape[2] += 2
        target_shape[3] += 2
        
        if self.scale_num > 0:
            if self.scale_num <= 3:
                dilation_val = self.opt.dil_val_small
            else:
                dilation_val = self.opt.dil_val_big
            
            reduction_rate = 1
        
            self.Inception_Block = InceptionBlock( ch_in = N, ch_out = N, spatial_dilation = dilation_val, sz_out = target_shape, reduction = reduction_rate)
        
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) 
        
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):
        x = self.head(x)
        
        if self.scale_num > 0:
            source_feat = x
        
        x = self.body(x)
        
        if self.scale_num > 0:
            target_feat = x
            x = self.Inception_Block(source_feat, target_feat)
        
        
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        
        if self.scale_num == 0:
            return_value = x + y
        else:
            return_value = x
        
        return return_value
