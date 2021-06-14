#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch
import torch.nn as nn


# In[91]:


class Block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size, stride, padding,activation="LeakyReLU",generator=False):
        super().__init__()
        if generator:
            self.x = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False)
        else:
            self.x = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        if activation =="Relu":
            self.activation = nn.ReLU()
        else:
            self.activation=nn.LeakyReLU(0.2)
    
    def forward(self,x):
        y = self.x(x)
        y = self.norm(y)
        y = self.activation(y)
        return y
        
        


# In[92]:


class Discriminator(nn.Module):
    def __init__(self,channels_img,features_d):
        super().__init__()
        self.disc = nn.Sequential( 
            nn.Conv2d(channels_img,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            Block(features_d,features_d*2,4,2,1),
            Block(features_d*2,features_d*4,4,2,1),
            Block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride = 2 , padding =0),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.disc(x)


# In[93]:


class Generator(nn.Module):
    def __init__(self,z_dims,channels_img,features_g):
        super().__init__()
        self.gen= nn.Sequential(
            Block(z_dims,features_g*16,4,1,0,"Relu",True),
            Block(features_g*16,features_g*8,4,2,1,"Relu",True),
            Block(features_g*8,features_g*4,4,2,1,"Relu",True),
            Block(features_g*4,features_g*2,4,2,1,"Relu",True),
            nn.ConvTranspose2d(features_g*2,channels_img,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
    
    def forward(self,x):
        return self.gen(x)
        
         


# In[94]:


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)


# In[95]:


def test():
    N,in_channels,H,W = 10,3,64,64
    z_dims = 100
    x= torch.randn(N,in_channels,H,W)
    disc = Discriminator(in_channels,8)
    initialize_weights(disc)
    print(disc(x).shape)
    gen = Generator(z_dims,in_channels,8)
    z=torch.randn((N,z_dims,1,1))
    print(gen(z).shape)


# In[96]:


test()


# In[ ]:




