from dataclasses import InitVar
from unet_model import UNet
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from cutout import CutOut
from torchvision.transforms import GaussianBlur

class InvariantInformationClusteringLoss(nn.Module):
    def __init__(self,device):
        super(InvariantInformationClusteringLoss, self).__init__()
        self.device = device

    def forward(self,y):

        #y = torch.softmax(y[:,unique_labels,:],1)
        EPS = 1e-2
        C = y.shape[1]
        B = y.shape[0]
        N = y.shape[2]
        yt = torch.permute(y,(0,2,1))

        P = torch.matmul(y*C/N,yt)

        #P = P/ torch.sum(P,-1,keepdim=True)

        P = P.clone()
    
        P[(P < EPS).data] = EPS

        #P = P[:,unique_labels,:]
        #print(P.shape)
        #P = P[:,:,unique_labels]
        #print(P.shape)
        #print(torch.max(y))
        #print(torch.mean(torch.sum(P,(1,2)),0))
        
        Pideal = torch.tile(torch.unsqueeze(torch.eye(C,dtype=torch.float32,device=self.device),0),[B,1,1])

        Pideal[(Pideal < EPS).data] = EPS

        #Pideal = Pideal[:,unique_labels,:]
        #Pideal = Pideal[:,:,unique_labels]

        #print('{0} {1}'.format(P.shape,Pideal.shape))
        loss = torch.mean(F.kl_div(P.log(),Pideal.log(),log_target=True,reduction='none'),dim=(1,2))
        return loss

class MaskModel(nn.Module):
    def __init__(self,device,num_classes,doCutout=False):
        super(MaskModel, self).__init__()
        self.device = device
        self.in_channels=3
        self.out_channels=3
        self.FVOUTDIM = 64
        self.num_classes = num_classes
        self.cutout = CutOut(device=device)
        self.doCutout = doCutout
        self.unet = UNet(self.in_channels,2,False)
        self.gn = nn.GroupNorm(3,3)#nn.BatchNorm2d(3)
        self.ycbcr2greyFunc = nn.Linear(1024,self.in_channels)

        self.iic = InvariantInformationClusteringLoss(device)
        self.iic_in = InvariantInformationClusteringLoss(device)
        self.F = 8
        self.Kr = 0.299
        self.Kg = 0.587
        self.Kb = 0.114
        self.rgb2ycbcrW = nn.Parameter(torch.reshape(torch.tensor([[self.Kr,self.Kg,self.Kb],\
                                                     [-(0.5)*(self.Kr/(1-self.Kb)),-(0.5)*(self.Kg/(1-self.Kb)),0.5],\
                                                     [0.5,-(0.5)*(self.Kg/(1-self.Kr)),-(0.5)*(self.Kb/(1-self.Kr))]],dtype=torch.float32),[3,3,1,1]),requires_grad=True)
        
        self.rgb2grey = nn.Parameter(torch.reshape(torch.tensor([0.2125,0.7154,0.0721],dtype=torch.float32),[1,3,1,1]),requires_grad=False)
        
        

    def pairwise_dist2 (self,A, B):  #A BxCxN1 B BxCxN2
        na = torch.sum(torch.square(A), 1) #BxN1
        nb = torch.sum(torch.square(B), 1) #BxN2

        # na 
        # as a row and nb as a co"lumn vectors
        na = na.reshape([A.shape[0], -1, 1]) #BxN1x1
        nb = nb.reshape([B.shape[0],1, -1]) #Bx1xN2

        # return pairwise euclidead difference matrix
        D = -(na - 2*torch.matmul(torch.permute(A,(0,2,1)), B) + nb)
        return D
    
    def pairwise_distance(self,x):
        inner = 2*torch.matmul(x.transpose(2, 1), x) #BxNxF x BxFxN
        xx = torch.sum(x**2, dim=1, keepdim=True)
        return torch.exp(-(xx - inner + xx.transpose(2, 1))/torch.var(x,dim=(1,2),keepdim=True))

    
    def forward(self, x):
        B = x.shape[0]

        xYcbcr = self.gn(F.conv2d(x,self.rgb2ycbcrW))

        yboth, coords, fv =self.unet(xYcbcr)

        yraw = yboth[:,0:2,:,:]
        
        if self.training and self.doCutout:
            mask = self.cutout(yraw[:,0:1,:,:])
            mask2channel = torch.cat((1-mask[:,0:1,:,:],(mask[:,0:1,:,:])),1)
        else:
            mask = torch.ones(yraw[:,0:1,:,:].shape,device=self.device,dtype=torch.float32)
            mask2channel = torch.ones(yraw.shape,device=self.device,dtype=torch.float32)

        y_sm = torch.softmax(yraw,1)

        lowClassMean = torch.mean(y_sm[:,0:1,:,:]*mask*coords,dim=(1,2,3))
        highClassMean = torch.mean(y_sm[:,1:2,:,:]*mask*coords,dim=(1,2,3))

        meansep_loss = torch.exp(-(lowClassMean - highClassMean)**2)
                
        iic_loss = self.iic(torch.reshape(y_sm*mask,[B,2,-1])) #masked pixels should be maximally uncertain
        loss =  iic_loss + meansep_loss

        return y_sm[:,1:2,:,:]*mask,loss
