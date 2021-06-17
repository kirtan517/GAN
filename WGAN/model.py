import torch 
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,generator = False):
        super().__init__()
        if generator:
            self.x = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding ,bias = False)
        else:
            self.x = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias = False)
        self.norm = nn.BatchNorm2d(out_channels)
        if generator:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.norm(self.x(x)))

class Discriminator(nn.Module):
    def __init__(self,in_channels,features_d):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels,features_d,kernel_size = 4 , stride =2, padding =1),
            nn.LeakyReLU(0.2),
            Block(features_d,features_d*2,4,2,1),
            Block(features_d*2,features_d*4,4,2,1),
            Block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size= 4,stride = 2 ,padding =0)
        )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,z_dim,in_channels,features_g):
        super().__init__()
        self.gen = nn.Sequential(
            Block(z_dim,features_g*16,4,1,0,True),
            Block(features_g*16,features_g*8,4,2,1,True),
            Block(features_g*8,features_g*4,4,2,1,True),
            Block(features_g*4,features_g*2,4,2,1,True),
            nn.ConvTranspose2d(features_g*2,in_channels,kernel_size = 4 , stride =2 ,padding =1)
            )
    def forward(self , x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if  isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.2)


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

if __name__ == "__main__":
    test()
