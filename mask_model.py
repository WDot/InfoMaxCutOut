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
        #self.doCutout = doCutout
        self.unet = UNet(self.in_channels,2 + 3,False)
        self.gn = nn.GroupNorm(3,3)#nn.BatchNorm2d(3)
        #self.ycbcr2greyFunc = nn.Linear(1024,self.in_channels)
        #https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf

        self.iic = InvariantInformationClusteringLoss(device)
        #self.iic_in = InvariantInformationClusteringLoss(device)
        self.F = 8
        self.Kr = 0.299
        self.Kg = 0.587
        self.Kb = 0.114
        self.rgb2grey = nn.Parameter(torch.reshape(torch.tensor(np.array([0.2125, 0.7154, 0.0721])),[1,3,1,1]),requires_grad=False)
        self.rgb2ycbcrW = nn.Parameter(torch.reshape(torch.tensor([[self.Kr,self.Kg,self.Kb],\
                                                     [-(0.5)*(self.Kr/(1-self.Kb)),-(0.5)*(self.Kg/(1-self.Kb)),0.5],\
                                                     [0.5,-(0.5)*(self.Kg/(1-self.Kr)),-(0.5)*(self.Kb/(1-self.Kr))]],dtype=torch.float32),[3,3,1,1]),requires_grad=True)
        
        #self.fv2weight = nn.Linear(1024,3)
        self.fv2fg = nn.Linear(1024,2)
        #self.eigenpredictor = nn.Linear(1024,1)
        
        

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
        return -(xx - inner + xx.transpose(2, 1))/(torch.var(x,dim=(1,2),keepdim=True)).detach()

    
    def forward(self, x,doCutout=False):
        B = x.shape[0]
        
        #xVals = torch.tile(torch.reshape(torch.linspace(0,1,x.shape[2],dtype=torch.float32,device=self.device),[1,1,x.shape[2],1]),[x.shape[0],1,1,x.shape[3]])
        #yVals = torch.tile(torch.reshape(torch.linspace(0,1,x.shape[3],dtype=torch.float32,device=self.device),[1,1,1,x.shape[3]]),[x.shape[0],1,x.shape[2],1])
        #xFull = torch.cat((x,xVals,yVals),dim=1)
        xYcbcr = self.gn(F.conv2d(x,self.rgb2ycbcrW))

        yboth, coords, fv =self.unet(xYcbcr)

        coords = coords.detach()

        #w_channel = torch.reshape(torch.softmax(self.fv2weight(fv),1),[B,-1,1,1])
        #coords = torch.softmax(coords,1)

        w = torch.reshape(torch.softmax(self.fv2fg(fv),1),[B,2,1,1])

        yraw = yboth[:,0:2,:,:]
        
        yrgb = torch.sigmoid(yboth[:,2:,:,:])
        
        #w_channel = torch.softmax(yboth[:,2:,:,:],1)
        
        if self.training and doCutout:
            mask = self.cutout(yraw[:,0:1,:,:])
            #metamask = torch.reshape((torch.rand([x.shape[0]],dtype=torch.float32,device=self.device) > 0.5),[-1,1,1,1]).type(torch.float32)
            #mask = mask*metamask + (1-metamask)*torch.ones(yraw[:,0:1,:,:].shape,device=self.device,dtype=torch.float32)
        else:
            mask = torch.ones(yraw[:,0:1,:,:].shape,device=self.device,dtype=torch.float32)
            #metamask = torch.reshape((torch.zeros([x.shape[0]],dtype=torch.float32,device=self.device)),[-1,1,1,1]).type(torch.float32)

        y_sm = torch.softmax(yraw,1)

        lowClassMean = torch.mean(y_sm[:,0:1,:,:]*mask*(coords),dim=(2,3))
        highClassMean = torch.mean(y_sm[:,1:2,:,:]*mask*(coords),dim=(2,3))

        means = torch.stack((lowClassMean,highClassMean),1)
        iic_means_loss = self.iic(means)
        #meansep_loss = torch.exp(-(lowClassMean - highClassMean)**2/torch.var(mask*w_channel*coords))
        #xYcbcrXy = torch.cat((xYcbcr,xVals,yVals),dim=1)*w_channel
        #xYcbcrSmall = F.avg_pool2d(coords*mask,self.F)
        #xYcbcrSmall = torch.reshape(xYcbcrSmall,[x.shape[0],xYcbcrSmall.shape[1],-1]) #1x2xHW/16
        #xYcbcrSmall = xYcbcrSmall / torch.sum(xYcbcrSmall,dim=2,keepdim=True).detach()

        #A = torch.sigmoid(self.pairwise_distance(xYcbcrSmall))

        #dcoeffs = torch.sum(A,dim=-1)

        #D = torch.diag_embed(dcoeffs,dim1=-2, dim2=-1)

        #print(dcoeffs)

        #L = (D - A)

        y_sm_detach = y_sm#.detach()

        #ysmall = F.avg_pool2d(y_sm[:,1:2,:,:]*mask,self.F)

        #eigenvalues = torch.reshape(self.eigenpredictor(fv),[-1,1,1])
        
        #y1d = torch.reshape(ysmall,[x.shape[0],1,-1]) #Bx1xHW/16
        #1dT = y1d.permute((0,2,1)) #BxHW/16x1

        #onesVec = torch.ones(y1dT.shape,dtype=torch.float32,device=self.device)
        #onesT = torch.ones(y1dT.shape,device='cuda')
        #onesT /= torch.sum(onesT,1,keepdim=True)

        #left = torch.matmul(L,y1dT)/(x.shape[2]*x.shape[3]/(self.F**2))
        #print('Y1D {0}'.format(torch.amin(y1d)))
        #right = eigenvalues * torch.matmul(D,y1dT)/(x.shape[2]*x.shape[3]/(self.F**2))
        #print('Y1D {0}'.format(torch.amin(y1d)))
        #eigen_loss = torch.mean((left - right)**2,dim=(1,2))

        #numerator = torch.matmul(torch.matmul(y1d,L)/(x.shape[2]*x.shape[3]/(self.F**2)),y1dT)#/(x.shape[2]*x.shape[3]/(self.F**2))**2
        
        #print('Y1D {0}'.format(torch.amin(y1d)))
        #denominator = torch.matmul(torch.matmul(y1d,D)/(x.shape[2]*x.shape[3]/(self.F**2)),y1dT) + 1e-3#/(x.shape[2]*x.shape[3]/(self.F**2)) + 1e-8
        #denominator = torch.matmul(y1d,y1dT) + 1e-3#/(x.shape[2]*x.shape[3]/(self.F**2)) + 1e-8

        #not_trivial = torch.mean(torch.matmul(y1d,onesT)/(x.shape[2]*x.shape[3]/(self.F**2)),(1,2))

        #constraint_loss = torch.mean(torch.matmul(torch.matmul(y1d,D)/(x.shape[2]*x.shape[3]/(self.F**2)),onesT),(1,2))
        #constraint_loss = torch.mean(torch.matmul(torch.matmul(y1d,D)/(x.shape[2]*x.shape[3]/(self.F**2)),onesVec),dim=(1,2))#/(x.shape[2]*x.shape[3]/(self.F**2))
        #constraint_loss = torch.mean(torch.matmul(y1d,onesVec),(1,2)) + 1e-
        output = (y_sm_detach[:,0:1,:,:]*w[:,0:1,:,:] + y_sm_detach[:,1:2,:,:]*w[:,1:2,:,:])
        yrgbmask = yrgb*output
        EPS = 1e-2
        xcopy = x.detach()*output
        xcopy[(xcopy < EPS).data] = EPS
        yrgbmask[(yrgb< EPS).data] = EPS
        reconstruction_loss = torch.mean(F.mse_loss(yrgbmask,xcopy,reduction='none'),dim=(1,2,3))
        #ncut_loss = torch.mean(numerator/denominator,(1,2)) + 1e-8
        iic_loss = self.iic(torch.reshape(y_sm*mask,[B,2,-1])) #masked pixels should be maximally uncertain
        iic_w_loss = self.iic(torch.permute(torch.reshape(w,[B,2,1]),(2,1,0)))
        loss = iic_w_loss + iic_means_loss + iic_loss + reconstruction_loss#iic_loss# + iic_means_loss + meansep_loss + not_trivial
        
        

        #print('{0} {1} {2} {3}'.format(torch.mean(numerator),torch.mean(denominator),torch.mean(ncut_loss), torch.mean(constraint_loss)))

        #*bimodalMask + (1-bimodalMask)*torch.sum(y_sm_detach,1,keepdim=True)
        #output = y_sm_detach[:,1:2,:,:]

        return output,mask,loss
