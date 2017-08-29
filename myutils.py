import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def colormap(n):
    cmap=np.zeros([n,3]).astype(np.uint8)
    for i in range(n):
        r,g,b=np.zeros(3)
        for j in range(8):
            r=r+(1<<(7-j))*((i&(1<<(3*j)))>>(3*j))
            g=g+(1<<(7-j))*((i&(1<<(3*j+1)))>>(3*j+1))
            b=b+(1<<(7-j))*((i&(1<<(3*j+2)))>>(3*j+2))
        cmap[i,:]=np.array([r,g,b])
    return cmap
class Relabel:
    def __init__(self,origin,newlabel):
        self.origin=origin
        self.newlabel=newlabel
    def __call__(self,tensor):
        assert isinstance(tensor,torch.LongTensor),"Tensor needs to be LongTensor"
        tensor[tensor==self.origin]=self.newlabel
        return tensor
class Tolabel:
    def __call__(self,image):
        return torch.from_numpy(np.array(image)).long().unsequeeze(0)
class Colorize:
    def __init__(self,n=21):
        self.cmap=colormap(256)
        self.cmap[n]=self.cmap[-1]
        self.cmap=torch.from_numpy(self.cmap[:n])
    def __call__(self,gray_image):
        size=gray_image.size()
        color_image=torch.ByteTensor(3,size[1],size[2]).fill_(0)
        for label in range(1,len(self.cmap)):
            mask=gray_image[0]==label
            color_image[0][mask]=self.cmap[label][0]
            color_image[1][mask]=self.cmap[label][1]
            color_image[2][mask]=self.cmap[label][2]
        return color_image
def imsave(file_name,img):
    assert type(img)==torch.FloatTensor,"img must be a torch.FloatTensor"
    n_dim=len(img.size())
    assert (n_dim==2 or n_dim==3,"img must be a 2 or 3 tensor")
    img=img.numpy()
    if n_dim==3:
        plt.imsave(file_name,np.transpose(img,(1,2,0)))
    else:
        plt.imsave(file_name,img,cmap="gray")

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss2d(weight)
    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)
