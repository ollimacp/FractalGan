#from ast import Param
import numpy as np
from collections import defaultdict

#Var for Line 17xx
found = None

#Helper-function to show any np.array as a picture with a chosen title and a colormapping 
def showNPArrayAsImage( np2ddArray, title, colormap, block):
    plt.figure()                    #Init figure
    plt.imshow(np2ddArray,          #Gererate a picture from np.array and add to figure
            interpolation='none',
            cmap = colormap)
    plt.title(title)                #Add title to figure
    plt.show(block=block)           #Show array as picture on screen, but dont block the programm to continue.
import sys
np.set_printoptions(threshold=sys.maxsize)




def showNPArrayImageArray(ArrayList, Namelist, opt, block):
    #source: https://stackoverflow.com/questions/22053274/grid-of-images-in-matplotlib-with-no-padding
    max_cols = opt.batch_size
    nrows = 3
    mul = 4 # Size multiplicator
    fig, axes = plt.subplots(nrows=nrows, ncols=max_cols, figsize=(int(opt.batch_size*mul),int(nrows*mul)))
    #lablelist = [labels_2[idx,0,:,:], labels_2[idx,1,:,:], labels_4[idx,0,:,:], labels_4[idx,1,:,:], labels_8[idx,0,:,:], labels_8[idx,1,:,:], labels_16[idx,0,:,:],  labels_16[idx,1,:,:], NumpyencImg2[idx,0,:,:], NumpyencImg2[idx,1,:,:], NumpyencImg4[idx,0,:,:], NumpyencImg4[idx,1,:,:], NumpyencImg8[idx,0,:,:], NumpyencImg8[idx,1,:,:], NumpyencImg16[idx,0,:,:]  , NumpyencImg16[idx,1,:,:]]
    lablelist = ArrayList
    #titlelist = ["BCR2", "LAC2", "BCR4", "LAC4", "BCR8", "LAC8","BCR16", "LAC16", "NN BCR2", "NN LAC2",   ]
    ylabellist = Namelist
    xlabellist = ["", "", "","", "", "","","","" ,"" ,"" ,"" ,"" ,"" ,"" ,"" ]
    #ylabellist = ["CPU", "", "","", "", "", "","", "GPU", "","","","", "", "",""]
    #xlabellist = ["", "", "","", "", "","","","BCR2" ,"LAC2" ,"BCR4" ,"LAC4" ,"BCR8" ,"LAC8" ,"BCR16" ,"LAC16" ]
    for idx, image in enumerate(lablelist):
        row = idx // max_cols
        col = idx % max_cols
        #axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
        #axes[row, col].set_title(titlelist[idx])
        # label x-axis and y-axis 
        axes[row, col].set_ylabel(ylabellist[idx]) 
        axes[row, col].set_xlabel(ylabellist[idx]) 

    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show(block=block)





def sample_image(n_row, batches_done,opt, input_imgs,inpainted_imgs, gen_imgs ):

    if sys.platform == "linux" or sys.platform == "darwin":
        saveplace = FileParentPath + "/data/Samples"+"/" +str(batches_done)+".png"
    elif sys.platform == "win32":
        saveplace = FileParentPath + "\\data\\Samples"+"\\" +str(batches_done)+".png"


    ArrayList = np.array([])
    Namelist = []
    #ori_imgs = real_imgs[:,0,:,:].detach().cpu().numpy() #batch, 1, y,x

    for i in range(opt.batch_size):

        #ArrayList = np.append(ArrayList,ori_imgs[i,:,:])

        if i == 0:
            Namelist.append("Input")
            ArrayList = np.array([input_imgs[i,:,:]])
        else:
            ArrayList = np.append(ArrayList,[input_imgs[i,:,:]],axis=0)
            Namelist.append("")

    for i in range(opt.batch_size):
        #ArrayList.append(inpainted_imgs[i,:,:])
        ArrayList = np.append(ArrayList,[inpainted_imgs[i,:,:]],axis=0)
        if i == 0:
            Namelist.append("Noise/Mask")
        else:
            Namelist.append("")


    for i in range(opt.batch_size):
        #ArrayList.append(gen_imgs[i,:,:])
        try:
            ArrayList = np.append(ArrayList,[gen_imgs[i,:,:]],axis=0)
        except:
            pass
        if i == 0:
            Namelist.append("Output")
        else:
            Namelist.append("")

    showNPArrayImageArray(ArrayList, Namelist, opt,False)


    #save_image(gen_imgs.data, saveplace , nrow=n_row, normalize=True)





def sample_latent_space(n_row, batches_done,opt, SBC):

    if sys.platform == "linux" or sys.platform == "darwin":
        saveplace = FileParentPath + "/data"+"/" +str(batches_done)+".png"
    elif sys.platform == "win32":
        saveplace = FileParentPath + "\\data"+"\\" +str(batches_done)+".png"

    """Saves a grid of generated digits"""
    # Sample noise
    #z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim**2))))
    z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, 1 ,opt.latent_dim, opt.latent_dim_x))))

    gen_imgs = decoder(z, real_labels_2, real_labels_4, real_labels_8, real_labels_16)
    #Numpydecoded_imgs =  gen_imgs.cpu()
    #Numpydecoded_imgs = Numpydecoded_imgs.detach().numpy()
    #img = Image.fromarray(Numpydecoded_imgs[0], 'L')
    #img.show()
    save_image(gen_imgs.data, saveplace , nrow=n_row, normalize=True)





try:
    import config
except:
    pass
from importlib import reload    # to reload config.py when file changes

'''
NOTES
use :
        for param in model.parameters():
            param.grad = None
 instead of model.zero_grad() or optimizer.zero_grad()



Many PyTorch APIs are intended for debugging and should be disabled for regular training runs:
anomaly detection: torch.autograd.detect_anomaly or torch.autograd.set_detect_anomaly(True)
profiler related: torch.autograd.profiler.emit_nvtx, torch.autograd.profiler.profile
autograd gradcheck: torch.autograd.gradcheck or torch.autograd.gradgradcheck

if device == cuda:
    torch.backends.cudnn.benchmark = True


https://stackoverflow.com/questions/58296345/convert-3d-tensor-to-4d-tensor-in-pytorch
x = torch.zeros((4,4,4))   # Create 3D tensor 
print(x[None].shape)       #  (1,4,4,4)
print(x[:,None,:,:].shape) #  (4,1,4,4)
print(x[:,:,None,:].shape) #  (4,4,1,4)
print(x[:,:,:,None].shape) #  (4,4,4,1)

train_data = train_data[:,None,:,:]
test_data = test_data[:,None,:,:]

print("train_data.shape", train_data.shape,"test_data.shape", test_data.shape)

GAUSSIAN NOISE FUNCTION

from https://github.com/ShivamShrirao/facegan_pytorch/blob/main/facegan_pytorch.ipynb

class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.001, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

###
def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins


#################################################

OR 

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 


#
DROPOUT LAYERS ############################################################

from https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network

In the original paper that proposed dropout layers, by Hinton (2012), dropout (with p=0.5) was used on each of the fully connected (dense) layers before the output; it was not used on the convolutional layers. This became the most commonly used configuration.

More recent research has shown some value in applying dropout also to convolutional layers, although at much lower levels: p=0.1 or 0.2. Dropout was used after the activation function of each convolutional layer: CONV->RELU->DROP




####################
BATCHNORM AFTER ACTIVATION FUNCTION

https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
###

'''

#BOXCOUNT INIT
Boxsize=[2,4,8,16,32,64,128]    #,256,512,1024]
#iteration = 0
from EvoNet import Network_Generator
import os
#os.makedirs("images", exist_ok=True)
#shape =  (2,2)

#Debugging tools----------------------------------------------------------------------------------
import linecache
import sys

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


#Imports------------------------------------------------------------------------------------------
#Importing nessecary modules for creating machine learning networks, such as 
import torch                                    #Pytorch machine learning framework.
import torch.nn as nn                           #
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim
#from torch.utils.data.sampler import SubsetRandomSampler

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
#import gc
#import math
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials  #hyperoptimization libary
from PIL import Image
#from os import listdir

# Common Directories
import pathlib              #Import pathlib to create a link to the directory where the file is at.
FileParentPath = str(pathlib.Path(__file__).parent.absolute())
saveplace = FileParentPath + "/Datasets/"

import itertools

Sample_layerdict ={'0.0078125': [1,1,128] ,'0.015625': [1,1,64] ,'0.03125': [1,1,32] ,'0.0625': [1,1,16] , '0.125': [1,1,8], '0.25':[1,1,4], '0.5':[1,1,2], '1.0':[1,1,1], '2.0':[2,2,1], '4.0':[4,4,1], '8.0':[8,8,1], '16.0':[16,16,1],'32.0':[32,32,1],'64.0':[64,64,1],'128.0':[128,128,1],'256.0':[256,256,1],}
#Setting device to GPU/CPU 
def get_device():
    global device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    return device

#device = get_device()
#print("------------------------------------ \n MANUAL SETTING TO CPU  FractalGANv6.py 121 \n -------------------------------------------")
#device ="cpu"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #To disable GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor


#taken from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params




#DEEPSPEED PLUGIN FOR MEMORY OFFLOADING


#from pytorch_lightning.plugins import DeepSpeedPlugin
#https://devblog.pytorchlightning.ai/accessible-multi-billion-parameter-model-training-with-pytorch-lightning-deepspeed-c9333ac3bb59
#import pytorch_lightning as pl
'''
trainer = pl.Trainer(
  plugins=DeepSpeedPlugin(
           stage=3))

#,           cpu_offload=True
'''



def reparameterization(mu, logvar,Tensor,opt):
    std = torch.exp(logvar / 2)
    #print("mu.shape",mu.shape)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size,1, opt.latent_dim, opt.latent_dim_x))))
    #print("reparameterization_shapes in Order: sampled_z, mu, logvar, std")
    reparameterization_shapes = [sampled_z.shape, mu.shape, logvar.shape, std.shape] 
    #print("reparameterization_shapes",reparameterization_shapes)
    z = sampled_z * std + mu
    return z



'''
NOISE FUNCTIONS
'''


# QUELLE ADD GAUSSIAN NOISE TO DIS TO IMPROVE LOSS
#https://github.com/ShivamShrirao/facegan_pytorch/blob/main/facegan_pytorch.ipynb
class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.001, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        #print("Gaussian noise input shape")
        #print(x.shape)
        if self.training:
            self.decay_step
            return x + torch.empty_like(x).normal_(mean = 0, std=self.std)

        else:
            return x



class mask_data(nn.Module):
    def __init__(self,opt ,maskmean, maskdimension, custom_mask = None):
        super(mask_data,self).__init__()

        self.custom_mask = custom_mask
        self.maskmean = maskmean
        self.maskmeanX , self.maskmeanY = maskmean
        
        self.maskdimension = maskdimension
        self.maskdimensionX, self.maskdimensionY = maskdimension
        
        self.maskminY = int(self.maskmeanY -self.maskdimensionY)
        self.maskminX = int(self.maskmeanX -self.maskdimensionX)

        self.maskmaxY = int(self.maskmeanY +self.maskdimensionY)
        self.maskmaxX = int(self.maskmeanX +self.maskdimensionX)

        #self.maskminX , self.maskminY = int(self.maskmeanX - self.maskdimensionX) ,  int(self.maskmeanY - self.maskdimensionY)
        #self.maskmaxX , self.maskmaxY = int(self.maskmeanX + self.maskdimensionX) ,  int(self.maskmeanY + self.maskdimensionY)
        
        print("Maskmean:"+ str(self.maskmean) , "Dimension Y min max"+ str(self.maskminY) +" "+ str(self.maskmaxY), "Dimension X min max: "+ str(self.maskminX) + str(self.maskmaxX))
    
    def forward(self,x):
        #generate a true/false mask where the borders apply # 4 dimensions [batch_size, channel, Y, X ]
        #print("inpainting x.shape is", x.shape)
        #print("Maskmean X/Y:"+ str(self.maskmean) , "Dimension Y min max"+ str(self.maskminY) +" "+ str(self.maskmaxY), "Dimension X min max: "+ str(self.maskminX) +" "+ str(self.maskmaxX))

        if self.custom_mask == None:
            #mask = x == x [:, :, self.maskminY:self.maskmaxY, self.maskminX:self.maskminX]+
            #create a zero map
            #print(opt.device)
            try:
                mask = torch.ones(x.shape,device=torch.device(opt.device))
            except:
                mask = torch.ones(x.shape,device=torch.device('cpu'))      #take everything mask
            #mask[:,:,self.maskminY:self.maskmaxY, self.maskminX:self.maskmaxX] = 0.0 #set mask in masked region to 0
            mask[:,:,self.maskminY:self.maskmaxY, self.maskminX:self.maskmaxX] = 0.0 #set mask in masked region to 0
            #mask = torch.zeros(x.shape,device= torch.device(opt.device))
            #print("Whole mask")
            #print(mask[0,0])
            #print("mask in masked region should be an array consisting out of 0.0")
            #print(mask[self.maskminY:self.maskmaxY, self.maskminX:self.maskmaxX])
        else:
            mask = self.custom_mask
        #mask = mask.int()  #convert True/False Mask into 1 and 0 to be able to multiply
        # to generate a mask in pytorch  source https://stackoverflow.com/questions/64764937/creating-a-pytorch-tensor-binary-mask-using-specific-values
        masked_result = torch.mul(x,mask)

        '''
        DEBUGGING SHOW ARRAYS of pic and BCRLAK
        numpymask = mask[0,0].detach().cpu().numpy()
        showNPArrayAsImage(numpymask,"mask","gray")

        maskedres = masked_result.detach().cpu().numpy()
        showNPArrayAsImage(maskedres[0,0], "Masked Data", "gray")  
        '''
        return masked_result



'''
This class is for altering the incoming pictures/Data into versions distorted for Neural inpainting or altering the aspect ratio.

THis class overlays the incoming data with noise for the network to learn denoising of blurred or grizzy pictures created with old hardware or noisy sources.

Also the data can be mask parially to enable neural inpainting for deleted parts by a process or a user.
The mask also can used to recreate aspect ratios like converting 4:3 images into 16:9 images by masking the original 16:9 image with borders into a 4:3 format for the network to repair


NOISE FUNCTIONS TAKING FROM : https://debuggercafe.com/adding-noise-to-image-data-for-deep-learning-data-augmentation/

'''
class Inpainting(nn.Module):
    def __init__(self,Parameter):
        super(Inpainting, self).__init__()
        '''
        THE inpainting Module masks the incoming data and the BCR/LAK accordingly so the network doesnt just
        cheat by taking unchanged BCR/LAK data, which it wouldnt have when getting noisy image and calc BCR then
        '''

        opt = Parameter['opt']
        self.randdommask = None

        ################### ###########################################################################################
        self.Layers = nn.ModuleList()

        superresolution = Parameter['superresolution']
        self.superres = superresolution[0]
        self.magnification = superresolution[1]
        
        if self.superres == True:
            #ORI DATA
            self.Layers.append(nn.AvgPool2d((self.magnification, self.magnification), stride=(self.magnification, self.magnification)))         #should be pool with poolkernel = stride, cause be reducing with 4 the poolkernel has to be 4 and the stride also 4
            self.Layers.append(nn.UpsamplingNearest2d(scale_factor=self.magnification))
            # BCR/LAC 2
            self.Layers.append(nn.AvgPool2d((self.magnification, self.magnification), stride=(self.magnification, self.magnification)))         #should be pool with poolkernel = stride, cause be reducing with 4 the poolkernel has to be 4 and the stride also 4
            self.Layers.append(nn.UpsamplingNearest2d(scale_factor=self.magnification))
            # BCR/LAC 4
            self.Layers.append(nn.AvgPool2d((self.magnification, self.magnification), stride=(self.magnification, self.magnification)))         #should be pool with poolkernel = stride, cause be reducing with 4 the poolkernel has to be 4 and the stride also 4
            self.Layers.append(nn.UpsamplingNearest2d(scale_factor=self.magnification))
            # BCR/LAC 8
            self.Layers.append(nn.AvgPool2d((self.magnification, self.magnification), stride=(self.magnification, self.magnification)))         #should be pool with poolkernel = stride, cause be reducing with 4 the poolkernel has to be 4 and the stride also 4
            self.Layers.append(nn.UpsamplingNearest2d(scale_factor=self.magnification))
            # BCR/LAC 16
            self.Layers.append(nn.AvgPool2d((self.magnification, self.magnification), stride=(self.magnification, self.magnification)))         #should be pool with poolkernel = stride, cause be reducing with 4 the poolkernel has to be 4 and the stride also 4
            self.Layers.append(nn.UpsamplingNearest2d(scale_factor=self.magnification))


        #For handling noisy images, the image can be overlayed with noise controlled by the noise parameter
        noise = Parameter['noise']
        
        self.noisebool = noise[0]
        self.std = noise[1]
        self.std_decay_rate = noise[2]
        if self.noisebool == True:
            self.Layers.append(  GaussianNoise(self.std, self.std_decay_rate) )
            #BCR/LAK 2
            self.Layers.append(GaussianNoise(self.std, self.std_decay_rate) )
            #BCR/LAK 4
            self.Layers.append(GaussianNoise(self.std, self.std_decay_rate) )
            #BCR/LAK 8
            self.Layers.append(GaussianNoise(self.std, self.std_decay_rate) )
            #BCR/LAK 16
            self.Layers.append(GaussianNoise(self.std, self.std_decay_rate) )


        '''
        if noise[0] == True:
            #Helperfunction to generate diffrent kind of noises. https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
            from skimage.util import random_noise

            def GaussianNoiseSkLearn(x):
                gauss_img = torch.tensor(random_noise(x, mode='gaussian', mean=0, var=0.05, clip=True))
                return gauss_img

            def SaltAndPepperNoise(x):
                SaltNPepperNoise = torch.tensor(random_noise(x, mode="s&p", salt_vs_pepper=0.5, clip=True ))
                return SaltNPepperNoise

            def SpeckleNoise(x):
                speckle_noise = torch.tensor(random_noise(x, mode='speckle', mean=0, var=0.05, clip=True))
                return speckle_noise
        '''
        
        ##############################################################################################################
        # taken from https://www.codefull.net/2020/03/masked-tensor-operations-in-pytorch/
        # to generate a mask in pytorch  source https://stackoverflow.com/questions/64764937/creating-a-pytorch-tensor-binary-mask-using-specific-values
  
        #For handling images with blacked out areas, the image can be overlayed with random or specific masks controlled by the mask parameter
        mask = Parameter['mask']
        #True/False, (x,y), (dx,dy)
        self.maskbool, self.maskmean, self.maskdimension = mask

        if self.maskbool == None:
            self.maskbool = True
            self.randdommask = True
            #TODO: SET MASKMEAN AND DIMENSION SUCH THAT FOR EVERY opt.img_size  accurate masks are chosen
            leftborder = int(opt.img_size[0] * 1/4)
            maxmasksize = leftborder
            rightborder = int(opt.img_size[0] * 3/4)
            self.maskmean = (torch.randint(leftborder,rightborder,(1,),device=torch.device(opt.device))[0]  , torch.randint(leftborder,rightborder,(1,),device=torch.device(opt.device))[0]  )
            self.maskdimension = (torch.randint(1,maxmasksize,(1,),device=torch.device(opt.device))[0]  , torch.randint(1,maxmasksize,(1,),device=torch.device(opt.device))[0]  )

        ori_maskmean, ori_maskdimension = self.maskmean, self.maskdimension

        if self.maskbool == True:
            self.Layers.append(mask_data( opt, self.maskmean, self.maskdimension, custom_mask = None ))
            ####BCRLAK2
            self.maskmean = self.maskmean[0]/2, self.maskmean[1]/2
            self.maskdimension = self.maskdimension[0]/2, self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmean, self.maskdimension, custom_mask = None ))
            ####BCRLAK4
            self.maskmean = self.maskmean[0]/2, self.maskmean[1]/2
            self.maskdimension = self.maskdimension[0]/2, self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmean, self.maskdimension, custom_mask = None ))
            ####BCRLAK8
            self.maskmean = self.maskmean[0]/2, self.maskmean[1]/2
            self.maskdimension = self.maskdimension[0]/2, self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmean, self.maskdimension, custom_mask = None ))
            ####BCRLAK16
            self.maskmean = self.maskmean[0]/2, self.maskmean[1]/2
            self.maskdimension = self.maskdimension[0]/2, self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmean, self.maskdimension, custom_mask = None ))
            #resetting maskmean & dimension
            self.maskmean, self.maskdimension = ori_maskmean, ori_maskdimension

        ##############################################################################################################

        #For handling images for aspect ratio transformation the incoming image can be overlayed with Letterbox(black bar at top and bottom), and Pillarbox (bar left and right) controlled by the parameters "Letterbox" and "Pillarbox"
        Letterbox, Pillarbox = Parameter['Letterbox'], Parameter['Pillarbox']
        
        self.LetterboxBool, self.LetterboxHeight = Letterbox
        if self.LetterboxBool == True:
            self.maskdimension  = ( opt.img_size[1],self.LetterboxHeight / 2 )
            # coordinates =              x                          ,  Y
            maskmeanUpperBox =    opt.img_size[1]/2, self.LetterboxHeight/2 
            maskmeanLowerBox =    opt.img_size[1]/2, opt.img_size[0] - self.maskdimension[0]
            self.Layers.append(mask_data(opt, maskmeanUpperBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, maskmeanLowerBox, self.maskdimension, custom_mask = None ))
        
            ####BCRLAK2
            maskmeanUpperBox = maskmeanUpperBox[0]/2, maskmeanUpperBox[1]/2 
            maskmeanLowerBox = maskmeanLowerBox[0]/2, maskmeanLowerBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, maskmeanUpperBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, maskmeanLowerBox, self.maskdimension, custom_mask = None ))
            ####BCRLAK4
            maskmeanUpperBox = maskmeanUpperBox[0]/2, maskmeanUpperBox[1]/2 
            maskmeanLowerBox = maskmeanLowerBox[0]/2, maskmeanLowerBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, maskmeanUpperBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, maskmeanLowerBox, self.maskdimension, custom_mask = None ))
            ####BCRLAK8
            maskmeanUpperBox = maskmeanUpperBox[0]/2, maskmeanUpperBox[1]/2 
            maskmeanLowerBox = maskmeanLowerBox[0]/2, maskmeanLowerBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, maskmeanUpperBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, maskmeanLowerBox, self.maskdimension, custom_mask = None ))
            ####BCRLAK16
            maskmeanUpperBox = maskmeanUpperBox[0]/2, maskmeanUpperBox[1]/2 
            maskmeanLowerBox = maskmeanLowerBox[0]/2, maskmeanLowerBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, maskmeanUpperBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, maskmeanLowerBox, self.maskdimension, custom_mask = None ))
            #resetting maskmean & dimension
            self.maskmean, self.maskdimension = ori_maskmean, ori_maskdimension


        
        ##############################################################################################################

        self.PillarboxBool, self.PillarboxWidth = Pillarbox

        if self.PillarboxBool == True:
            #create mask tensor
            self.maskdimension  =  (opt.img_size[0] , self.PillarboxWidth / 2 )
            # coordinates =  y                                      ,   x
            self.maskmeanLeftBox =   opt.img_size[0]/2                   ,  self.maskdimension[1]
            self.maskmeanRightBox = opt.img_size[0]/2                    ,  opt.img_size[1] - self.maskdimension[1]

            self.Layers.append(mask_data(opt, self.maskmeanLeftBox , self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, self.maskmeanRightBox, self.maskdimension, custom_mask = None ))


            ####BCRLAK2
            self.maskmeanLeftBox = self.maskmeanLeftBox[0]/2, self.maskmeanLeftBox[1]/2
            self.maskmeanRightBox = self.maskmeanRightBox[0]/2, self.maskmeanRightBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmeanLeftBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, self.maskmeanRightBox, self.maskdimension, custom_mask = None ))
            ####BCRLAK4
            self.maskmeanLeftBox = self.maskmeanLeftBox[0]/2, self.maskmeanLeftBox[1]/2
            self.maskmeanRightBox = self.maskmeanRightBox[0]/2, self.maskmeanRightBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmeanLeftBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, self.maskmeanRightBox, self.maskdimension, custom_mask = None ))
            ####BCRLAK8
            self.maskmeanLeftBox = self.maskmeanLeftBox[0]/2, self.maskmeanLeftBox[1]/2
            self.maskmeanRightBox = self.maskmeanRightBox[0]/2, self.maskmeanRightBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmeanLeftBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, self.maskmeanRightBox, self.maskdimension, custom_mask = None ))
            ####BCRLAK16
            self.maskmeanLeftBox = self.maskmeanLeftBox[0]/2, self.maskmeanLeftBox[1]/2
            self.maskmeanRightBox = self.maskmeanRightBox[0]/2, self.maskmeanRightBox[1]/2
            self.maskdimension = self.maskdimension[0]/2,self.maskdimension[1]/2
            self.Layers.append(mask_data(opt, self.maskmeanLeftBox, self.maskdimension, custom_mask = None ))
            self.Layers.append(mask_data(opt, self.maskmeanRightBox, self.maskdimension, custom_mask = None ))
            #resetting maskmean & dimension
            self.maskmean, self.maskdimension = ori_maskmean, ori_maskdimension

        print("MaskingStructure")
        print(self.Layers)


        ##############################################################################################################

    def forward(self,x, BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16):
        for i in range(len(self.Layers)):
            layerindex = 0



            if self.superres == True:
                #ORI DATA 
                x = self.Layers[layerindex](x)
                layerindex +=1
                x = self.Layers[layerindex](x)
                layerindex +=1
                # BCR/LAC 2
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1
                # BCR/LAC 4                
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                #BCR/LAC 8
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                #BCR /LAC 16
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1


            if self.noisebool == True:
                #print("Adding Noise,now")
                #layerindex = 0
                x = self.Layers[layerindex](x)
                layerindex +=1
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1

            if self.maskbool == True:
                #Print Masking
                x = self.Layers[layerindex](x)
                layerindex +=1
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1

            if self.LetterboxBool == True:
                #print("CREATE LETTERBOX")
            
                #For UPPER AND LOWER  respectively
                x = self.Layers[layerindex](x)
                layerindex +=1
                x = self.Layers[layerindex](x)
                layerindex +=1                
                #BCRLAK2
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1                
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1
                #BCRLAK4
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                
                #BCRLAK8
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                #BCRLAK16
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1
                
                

            if self.PillarboxBool == True:
                #print("CREATE PillarBOX")
            
                #For LEFT AND RIGHT  respectively
                x = self.Layers[layerindex](x)
                layerindex +=1
                x = self.Layers[layerindex](x)
                layerindex +=1                
                #BCRLAK2
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1                
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1
                #BCRLAK4
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                
                #BCRLAK8
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                #BCRLAK16
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1




                x = self.Layers[layerindex](x)
                layerindex +=1
                BCR_LAK_map_2 = self.Layers[layerindex](BCR_LAK_map_2)
                layerindex +=1
                BCR_LAK_map_4 = self.Layers[layerindex](BCR_LAK_map_4)
                layerindex +=1
                BCR_LAK_map_8 = self.Layers[layerindex](BCR_LAK_map_8)
                layerindex +=1
                BCR_LAK_map_16 = self.Layers[layerindex](BCR_LAK_map_16)
                layerindex +=1

        return x, BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16



# deprecheated n_size = 0      #The size of the input/Output Neurons of a Linar Layer connecting to an 4D Conv layer... has to be calulated by passing an input batch and calc the size after passing through
#Sample_layerdict ={'0.0078125': [1,1,128] ,'0.015625': [1,1,64] ,'0.03125': [1,1,32] ,'0.0625': [1,1,16] , '0.125': [1,1,8], '0.25':[1,1,4], '0.5':[1,1,2], '1.0':[1,1,1], '2.0':[2,2,1], '4.0':[4,4,1], '8.0':[8,8,1], '16.0':[16,16,1],'32.0':[32,32,1],'64.0':[64,64,1],'128.0':[128,128,1],'256.0':[256,256,1],}

class Encoder(nn.Module):
    def __init__(self,Parameter):
        super(Encoder, self).__init__()
        print("-------------INITIALIZE ENCODER----------------")
        #global n_size
        #The Encoder gets an 2d np array imagelike or MPM'd aud,vid etc; and converts it to the latent space
        # -----------------------------------------------------------------------------
        self.input_shape = Parameter['input_shape']
        print("self.input_shape",self.input_shape)
        self.LayerDescription = Parameter['LayerDescription']
        self.LayerCount = len(self.LayerDescription)
        self.Layers = nn.ModuleList()
        self.magnification = Parameter['magnification']
        #self.BoxcountEncoder = Parameter['SpacialBoxcounting']
        self.opt = Parameter['opt']
        #print(self.opt)
        #self.No_latent_spaces = self.opt.No_latent_spaces
        self.device = Parameter['device']
        self.InterceptLayerMagnification = np.array([0.5,0.25,0.125,0.0625])
        self.AggregateMagnification = 1.0       # to Calc, the magnification until from beginning until the present layer
        self.resultedIndexes = []

        for i in range(self.LayerCount):        #iterate over Layers 
            if i > 0:
                #Cause the input of a channel has to be the output of the last channels, adjust the input channels accordingly by multiply the previous output channels of the last layers
                previous_Output_channels = OUT
                #print("previous_Output_channels",previous_Output_channels)
                previous_parallel_layer_count = len(parallel_layers)
                #print("previous_parallel_layer_count",previous_parallel_layer_count)

            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = self.LayerDescription[i]
            #print("Layer",i," with Parameters layermagn, parallel_layers , channellist, batchnorm_switch", layermagn, parallel_layers , channellist, batchnorm_switch)
            self.AggregateMagnification = self.AggregateMagnification*float(layermagn)
            
            #print("InterceptLayerMagnification", self.InterceptLayerMagnification)
            #print("AggregateMagnification", self.AggregateMagnification)

            #Show me the indexes where the boxcounts have to be merged with the NN Layers
            result = np.where( self.InterceptLayerMagnification == self.AggregateMagnification)
            try:
                converted_index = int(result[0])
                #print("Converted_index", converted_index)
                self.resultedIndexes.append([i, converted_index])
                #print("Resulted index describes: [Index in Layercount, index for BC/LAC")
            except:
                PrintException()

            IN, OUT = channellist
            if i > 0:
                IN = previous_Output_channels * previous_parallel_layer_count
            
            layer_multiplicator =  np.array(Sample_layerdict[layermagn])   #multiplicator for each layer, so that with every chosen kernelsize the stride and pooling is scaled accordingly
            
            for parallel_layer in parallel_layers:

                if self.opt.residual == True:
                    output_size = int(float(self.opt.img_size[0]) * self.AggregateMagnification)
                    #print("residual output size for adaptive pool to pass onto the next layer", output_size)
                    self.Layers.append(nn.AdaptiveAvgPool2d((output_size,output_size)))

                if gaussian_noise == 1:
                    std = 0.001     #moderate disturbance
                    std_decay_rate = 0
                    self.Layers.append(  GaussianNoise(std, std_decay_rate) )

                kernel, stride, pool = layer_multiplicator * float(parallel_layer)
                kernel, stride, pool = int(kernel), int(stride), int(pool)
                #for the moment square but later can be implemented in x-y manner 
                Kx, Ky = kernel, kernel
                Sx, Sy = stride, stride
                Px,Py = 0,0     #no padding will be needed

                #New Pool version with adaptive average pooling
                output_size = int(float(self.opt.img_size[0]) * self.AggregateMagnification)
                print("output size for adaptive pool", output_size)
                

                if batchnorm_switch == 1:
                    #if batchnorm is applied, then bias calc of conv layers is not needed -> performance gain, less mem usage
                    self.Layers.append(nn.ConvTranspose2d(IN, OUT , kernel_size=(Kx, Ky), stride=(Sx, Sy), output_padding=(Px, Py), bias = False )) 
                else:
                    self.Layers.append(nn.ConvTranspose2d(IN, OUT , kernel_size=(Kx, Ky), stride=(Sx, Sy), output_padding=(Px, Py) )) 

                self.Layers.append(nn.AdaptiveAvgPool2d((output_size,output_size)))
                #self.Layers.append(nn.AvgPool2d((pool, pool), stride=(pool, pool)))        #Old version of pooling caused pictures looking blocky 

                # cause MaxPool(Relu(x)) = Relu(MaxPool(x)) the activation is applied after the pooling, cause less parameters have to be activated -> proof of performance pending
                self.Layers.append(nn.LeakyReLU(inplace = True))

                if dropout[0] == 1:
                    #if dropout switch is 1, then add dropout to layers
                    p = dropout[1] #with percentage of ...
                    self.Layers.append(nn.Dropout2d(p=p))

                if batchnorm_switch == 1:
                    self.Layers.append( nn.BatchNorm2d(OUT, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) )


        previous_Output_channels = OUT
        #print(f"Last Output of inception layer_channels {previous_Output_channels}")
        previous_parallel_layer_count = len(parallel_layers)
        #print(f"previous_parallel_layer_count {previous_parallel_layer_count}")
        IN = previous_Output_channels * previous_parallel_layer_count
        
        #self.Layers.append(nn.Conv2d(IN,No_latent_spaces, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) )      #1x1 channel reduction layer
        # a kernelsize of 2 with stride 1 results in a mag of 1/2 , 
        #print(f"CALCING NEEDED POOLSIZE  = int(self.AggregateMagnification/(self.opt.latent_dim/ self.opt.img_size[0])) = {self.AggregateMagnification}/({self.opt.latent_dim}/{self.opt.img_size[0]})")
        neededpoolsize_y = int(self.AggregateMagnification/(self.opt.latent_dim/ self.opt.img_size[0]))
        neededpoolsize_x = int(self.AggregateMagnification/(self.opt.latent_dim_x/ self.opt.img_size[1]))

        #print("reparametrization trick with cnn architecture")
        self.mu = nn.Conv2d(IN,self.opt.No_latent_spaces, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.logvar = nn.Conv2d(IN,self.opt.No_latent_spaces, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.reparamPool = nn.AvgPool2d((neededpoolsize_y, neededpoolsize_x), stride=(neededpoolsize_y, neededpoolsize_x))          #should be pool with poolkernel = stride, cause be reducing with 4 the poolkernel has to be 4 and the stride also 4
    
        self.Tensor = torch.cuda.FloatTensor if self.device == "cuda" else torch.FloatTensor

        print("Encoder structure")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"Input Size: {self.opt.img_size}")
        print(f"output/latent Size: {self.opt.img_size[0] * self.AggregateMagnification}")

        print(f"self.AggregateMagnification {self.AggregateMagnification}")
        print(f"self.opt.latent_dim {self.opt.latent_dim}")
        #print(f"self.opt.latent_dim_x{self.opt.latent_dim_x}")
        print(f"self.opt.img_size {self.opt.img_size}")
        print(" ENCODER : Resulted layer indexes, where BCR/LAC arrays have to be passed to are: "+ str(self.resultedIndexes))

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(self.Layers)
        print("Reparameterization structure")
        print("mu",self.mu)
        print("logvar",self.logvar)
        print("reparamPool",self.reparamPool)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("-------------INITIALIZE ENCODER COMPLETE----------------")




    def forward(self, x, BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16):
        #print("FORWARDING ENCODER--------------------------------")
        layerindex = 0
        
        for i in range(self.LayerCount):        #iterate forwards 
            #layermagn, parallel_layers , channellist, batchnorm_switch = self.LayerDescription[i]
            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = self.LayerDescription[i]
            
            data_branch = [None] * len(parallel_layers)      # cause single datapackage floods into the neurons, but cause of inception, multiple outputs 
            passed_data = x  #reshape data into  nd tensor

            for  index, parallel_layer in enumerate(parallel_layers):

                if self.opt.residual == True:
                    residual = self.Layers[layerindex](x)
                    layerindex +=1 

                if gaussian_noise == 1:
                    data_branch[index] = self.Layers[layerindex](passed_data)  #gaussian noise
                    #print(self.Layers[layerindex])
                    layerindex +=1 
                    data_branch[index] = self.Layers[layerindex](data_branch[index])
                else:
                    data_branch[index] = self.Layers[layerindex](passed_data)      #convlayer

                #print(self.Layers[layerindex])
                layerindex +=1
                data_branch[index] = self.Layers[layerindex](data_branch[index])   # pool layer
                #print(self.Layers[layerindex])
                layerindex +=1
                data_branch[index] = self.Layers[layerindex](data_branch[index])   # activation layer
                #print(self.Layers[layerindex])
                layerindex+=1

                if dropout[0] == 1:
                    data_branch[index] = self.Layers[layerindex](data_branch[index]) #dropout layer
                    #print(self.Layers[layerindex])
                    layerindex +=1

                if batchnorm_switch == 1:
                    data_branch[index] = self.Layers[layerindex](data_branch[index]) # batchnorm layer
                    #print(self.Layers[layerindex])
                    layerindex +=1

                #if its the first element of the paralelle layers just inherent x and after that concatenate the channels, cause dimensions are the same in x,y batchsize
                if index == 0:
                    x = data_branch[index]
                    #print("x.shape for intercept layer",x.shape)
                    #------------------------------------------------------
                    #SBC SHOULD BE INTERCEPTED HERE
                    #SBC = [BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16]
                    #self.resultedIndexes = [[1,0],[2,1],[3,3]]  [layercountindex,bcrindex]
                    #print("SBC[0].shape", SBC[0].shape)
                    #print("SBC[0] tpye", type(SBC[0]))
                    #print("SBC[0] dtype", SBC[0].dtype)
                    #------------------------------------------------------
                    #IF the Layercount matches the resulted indexes, then the adding can happen
                    for layercountindex, bcrindex in self.resultedIndexes:
                        if i == layercountindex:
                            #needed_idx = bcrindex
                            if bcrindex == 0:
                                BCR_LAC = BCR_LAK_map_2
                            elif bcrindex == 1:
                                BCR_LAC = BCR_LAK_map_4
                            elif bcrindex == 2:
                                BCR_LAC = BCR_LAK_map_8
                            elif bcrindex == 3:
                                BCR_LAC = BCR_LAK_map_16
                            if device == "cuda":
                                BCR_LAC.cuda()

                            #BCR_LAC = SBC[bcrindex]  #getting the correct shaped BCR and LAC
                            #print("BCR_LAC",BCR_LAC.shape)
                            #print("type x", type(x))
                            #print("  x.shape", x.shape)

                            merged_channels = torch.add(BCR_LAC, x[:,:2,:,:] )
                            merged_channels = torch.div(merged_channels,2)
                            #print("merged_channels",merged_channels)
                            #print("  merged_channels.shape", merged_channels.shape)
                            #print("x[:,2:,:,:].shape", x[:,2:,:,:].shape)
                            x = torch.cat(( merged_channels  , x[:,2:,:,:]),1)
                            #TODO: IF torch.cat is applied before relu OR JUST APPLY A ACTIVATION FUNCTION AFTER CAT bcr+residual to stay below 1. then we dont need torch.div 2
                    
                    #print("x.shape", x.shape)
                    #------------------------------------------------------
                else:
                    
                    x = torch.cat((x,data_branch[index]),1)


                if self.opt.residual == True:
                    residual_channellenght = int(residual.shape[1])
                    output_channellength = int(x.shape[1])
                    # if channels in layer before are more than now, just trash the last channels from past layer
                    if residual_channellenght > output_channellength:
                        residual_channellenght = output_channellength

                    passed_residual = torch.add(residual[:,:residual_channellenght,:,:], x[:,:residual_channellenght,:,:] )
                    passed_residual = torch.div(passed_residual,2)
                    #print("passed_residual",passed_residual)
                    #print("  passed_residual.shape", passed_residual.shape)
                    #print("x[:,2:,:,:].shape", x[:,2:,:,:].shape)
                    x = torch.cat(( passed_residual  , x[:,residual_channellenght:,:,:]),1)

        #x = torch.flatten(x,start_dim=1)
        #print("EncoderOutput flattend for  reparameterization x.shape",x.shape)  
        #print("IMG Compressed size")
        #print(x.size())
        mu = self.mu(x)
        mu = self.reparamPool(mu)
        logvar = self.logvar(x)
        logvar = self.reparamPool(logvar)
        z = reparameterization(mu, logvar, self.Tensor,self.opt)
        #print("z.shape Encoder Output shape latent dim",z.shape)  
        #z = torch.reshape(z,(opt.batch_size,1,opt.latent_dim,opt.latent_dim_x))
        # print("z.shape Encoder Output shape latent dim",z.shape)  

        return z



class Decoder(nn.Module):

    def __init__(self,Parameter):
        super(Decoder, self).__init__()
        print("--------INITIALIZE Generator/DECODER----------------")
        self.LayerDescription = Parameter['LayerDescription']
        self.LayerCount = len(self.LayerDescription)
        self.Layers = nn.ModuleList()
        self.magnification = Parameter['magnification']  # kann evtl raus
        self.opt = Parameter ['opt']

        '''
        LayerDescription = [     gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch 
                                    [1, '1', [1]          , [1,4]  , [1, 0.112], 0 ],
                                    [1, '2', [1,2,4,8]    , [4,8]  , [1, 0.112], 1 ],
                                    [0, '2', [1,2,16]     , [8,16] , [1, 0.112], 1 ],
                                    [0, '4', [1,2,4,8,16] , [16,32], [1, 0.112], 1 ],
                                    [1, '1', [1,2,4,8,16] , [32,1] , [1, 0.112], 0 ],
                                ]         
        
        '''
        
        self.InterceptLayerMagnification = np.array([ self.opt.img_size[0]/16.0 , self.opt.img_size[0]/8.0 , self.opt.img_size[0]/4.0  , self.opt.img_size[0]/2.0])

        self.AggregateMagnification = self.opt.latent_dim
        self.resultedIndexes = []

        for i in range(self.LayerCount): 
            if i > 0:
                #Cause the input of a channel has to be the output of the last channels, 
                #adjust the input channels accordingly by multiply the previous output channels of the last layers
                previous_Output_channels = OUT
                #print("previous_Output_channels",previous_Output_channels)
                previous_parallel_layer_count = len(parallel_layers)
                #print("previous_parallel_layer_count",previous_parallel_layer_count)

            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = self.LayerDescription[i]
            self.AggregateMagnification = self.AggregateMagnification*float(layermagn)
            
            print("Magnifications, where BCR/LAC are added to the network", self.InterceptLayerMagnification)
            #print("AggregateMagnification", self.AggregateMagnification)
            #print("layermag", layermagn)
            result = np.where( self.InterceptLayerMagnification == self.AggregateMagnification)        
            
            try:
                converted_index = 3- int(result[0]) #backwards
                print("Converted_index", converted_index)
                self.resultedIndexes.append([i, converted_index])
            except:
                PrintException()
            #print("resultedIndex where residual has to be passed to", self.resultedIndexes)

            IN, OUT = channellist
            if i > 0:
                IN = previous_Output_channels * previous_parallel_layer_count

            layer_multiplicator =  np.array(Sample_layerdict[layermagn])   #multiplicator for each layer, so that with every chosen kernelsizea and the stride is scaled accordingly
            
            for parallel_layer in parallel_layers:

                if self.opt.residual == True:
                    output_size = int( self.AggregateMagnification)
                    #print("residual output size for adaptive pool to pass onto the next layer", output_size)
                    self.Layers.append(nn.AdaptiveAvgPool2d((output_size,output_size)))
               
                if gaussian_noise == 1:
                    std = 0.001     #moderate disturbance
                    std_decay_rate = 0
                    self.Layers.append(  GaussianNoise(std, std_decay_rate) )

                kernel, stride, pool = layer_multiplicator * float(parallel_layer)
                kernel, stride, pool = int(kernel), int(stride), int(pool)
                #for the moment square but later can be implemented in x-y manner 
                Kx, Ky = kernel, kernel
                Sx, Sy = stride, stride
                Px,Py = 0,0     #no padding will be needed

                if batchnorm_switch == 1:
                    #if batchnorm is applied, then bias calc of conv layers is not needed -> performance gain
                    self.Layers.append(nn.ConvTranspose2d(IN, OUT , kernel_size=(Kx, Ky), stride=(Sx, Sy), output_padding=(Px, Py), bias = False )) #,output_padding = ()     #Attention ENhancer IN OUT Switched
                else:
                    self.Layers.append(nn.ConvTranspose2d(IN, OUT , kernel_size=(Kx, Ky), stride=(Sx, Sy), output_padding=(Px, Py) )) #,output_padding = ()     #Attention ENhancer IN OUT Switched

                output_size = int(self.AggregateMagnification)
                #print("output size for adaptive pool", output_size)
                self.Layers.append(nn.AdaptiveAvgPool2d((output_size,output_size)))
                #self.Layers.append(nn.AvgPool2d((pool, pool), stride=(pool, pool)))         #old version of pooling caused blocky pixealted output

                # cause MaxPool(Relu(x)) = Relu(MaxPool(x)) the activation is applied after the pooling, cause less parameters have to be activated -> proof of perf. is pending
                self.Layers.append(nn.LeakyReLU(inplace = True))

                if dropout[0] == 1:
                    #if dropout switch is 1, then add dropout to layers
                    p = dropout[1] #with percentage of ...
                    self.Layers.append(nn.Dropout2d(p=p))

                if batchnorm_switch == 1:
                    self.Layers.append( nn.BatchNorm2d(OUT, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) )

        #print(" DECODER : resultedIndex where box counts has to be passed to", self.resultedIndexes)
        print("Decoder/Generator")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"Input Size: {self.opt.latent_dim}")
        print(f"output Size: {self.AggregateMagnification}")


        print(f"self.opt.latent_dim {self.opt.latent_dim}")
        #print(f"self.opt.latent_dim_x{self.opt.latent_dim_x}")
        print(f"self.opt.img_size {self.opt.img_size}")
        print(" DECODER : Resulted layer indexes, where BCR/LAC arrays have to be passed to are: "+ str(self.resultedIndexes))

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(self.Layers)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("--------INITIALIZING DECODER DONE----------------")

    def forward(self, x, BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16):

        layerindex = 0
        for i in range(self.LayerCount):        #iterate forwards 

            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = self.LayerDescription[i]            
            #print("layerindex",i, 'Layermagnification', layermagn)

            data_branch = [None] * len(parallel_layers) 
            passed_data = x  

            for  index, parallel_layer in enumerate(parallel_layers):
                #print("parallel layer index", index)

                if self.opt.residual == True:
                    residual = self.Layers[layerindex](x)
                    layerindex +=1 

                if gaussian_noise == 1:
                    #data_branch[index] = self.Layers[layerindex](data_branch[index])  #gaussian noise
                    data_branch[index] = self.Layers[layerindex](passed_data)
                    #print(self.Layers[layerindex])

                    layerindex +=1 
                    data_branch[index] = self.Layers[layerindex](data_branch[index])
                else:
                    data_branch[index] = self.Layers[layerindex](passed_data)      #convlayer

                #print(self.Layers[layerindex])
                layerindex +=1
                data_branch[index] = self.Layers[layerindex](data_branch[index])   # pool layer
                #print(self.Layers[layerindex])
                layerindex +=1
                data_branch[index] = self.Layers[layerindex](data_branch[index])   # activation layer
                #print(self.Layers[layerindex])
                layerindex+=1
                
                if dropout[0] == 1:
                    #p = dropout[1]
                    data_branch[index] = self.Layers[layerindex](data_branch[index]) #dropout layer
                    #print(self.Layers[layerindex])
                    layerindex +=1

                if batchnorm_switch == 1:
                    data_branch[index] = self.Layers[layerindex](data_branch[index])
                    #print(self.Layers[layerindex])
                    layerindex +=1

                #if its the first element of the paralelle layers just inherent x and after that concatenate the channels, cause dimensions are the same in x,y batchsize
                if index == 0:
                    x = data_branch[index]
                    #print("x.shape for intercept layer",x.shape)

                    #------------------------------------------------------
                    #SBC = [BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16]
                    #self.resultedIndexes = [[1,0],[2,1],[3,3]]  [layercountindex,bcrindex]
                    #print("SBC[0].shape", SBC[0].shape)
                    #print("SBC[0] tpye", type(SBC[0]))
                    #print("SBC[0] dtype", SBC[0].dtype)
                    
                    #IF the Layercount matches the resulted indexes, then the adding can happen
                    for layercountindex, bcrindex in self.resultedIndexes:
                        if i == layercountindex:
                            #print("i == layercountidex", i)
                            #print("self.resulted INdexes",self.resultedIndexes)
                            needed_idx = bcrindex
                            
                            #other way around cause from little to big
                            if needed_idx == 0:
                                BCR_LAC = BCR_LAK_map_2
                                #print("bcr 2")
                            elif needed_idx == 1:
                                #print("bcr 4")
                                BCR_LAC = BCR_LAK_map_4
                            elif needed_idx == 2:
                                #print("bcr 8")
                                BCR_LAC = BCR_LAK_map_8
                            elif needed_idx == 3:
                                #print("bcr 16")
                                BCR_LAC = BCR_LAK_map_16

                            if device == "cuda":
                                BCR_LAC.cuda()


                            merged_channels = torch.add(BCR_LAC, x[:,:2,:,:] )
                            merged_channels = torch.div(merged_channels,2)
                            #print("merged_channels",merged_channels)
                            #print("  merged_channels.shape", merged_channels.shape)
                            #print("x[:,2:,:,:].shape", x[:,2:,:,:].shape)
                            x = torch.cat((merged_channels,x[:,2:,:,:]),1)
                    
                            #print("sleeping  BCR_LAC.shape", BCR_LAC.shape)
                else:
                    x = torch.cat((x,data_branch[index]),1)


                if self.opt.residual == True:
                    residual_channellenght = int(residual.shape[1])
                    output_channellength = int(x.shape[1])
                    # if channels in layer before are more than now, just trash the last channels from past layer
                    if residual_channellenght > output_channellength:
                        residual_channellenght = output_channellength

                    passed_residual = torch.add(residual[:,:residual_channellenght,:,:], x[:,:residual_channellenght,:,:] )
                    passed_residual = torch.div(passed_residual,2)
                    #print("passed_residual",passed_residual)
                    #print("  passed_residual.shape", passed_residual.shape)
                    #print("x[:,2:,:,:].shape", x[:,2:,:,:].shape)
                    x = torch.cat(( passed_residual  , x[:,residual_channellenght:,:,:]),1)

        return x


class Discriminator(nn.Module):
    def __init__(self,Parameter):
        super(Discriminator, self).__init__()

        print("--------------INITIALIZING Discriminator----------------")
        # -----------------------------------------------------------------------------
        self.input_shape = Parameter['input_shape']
        self.LayerDescription = Parameter['LayerDescription']
        self.LayerCount = len(self.LayerDescription)
        self.Layers = nn.ModuleList()
        self.opt = Parameter['opt']
        self.device = Parameter['device']
        self.AggregateMagnification = float(self.opt.latent_dim)

        for i in range(self.LayerCount):        #iterate over layers
            # if not the first layer...
            if i > 0:
                #Cause the input of a channel has to be the output of the last channels, 
                #adjust the input channels accordingly by multiply the previous output channels of the last layers
                previous_Output_channels = OUT
                #print("previous_Output_channels",previous_Output_channels)
                previous_parallel_layer_count = len(parallel_layers)
                #print("previous_parallel_layer_count",previous_parallel_layer_count)

            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch  = self.LayerDescription[i]
            self.AggregateMagnification = self.AggregateMagnification*float(layermagn)
            
            #print("AggregateMagnification", self.AggregateMagnification)
            #print("layermag", layermagn)

            IN, OUT = channellist
            if i > 0:
                IN = previous_Output_channels * previous_parallel_layer_count

            layer_multiplicator =  np.array(Sample_layerdict[layermagn])   #multiplicator for each layer, so that with every chosen kernelsize the stride and pooling is scaled accordingly
            
            for parallel_layer in parallel_layers:
            
                if self.opt.residual == True:
                    #output_size = int(float(opt.img_size[0]) * self.AggregateMagnification)
                    output_size = int( self.AggregateMagnification)
                    #print("residual output size for adaptive pool to pass onto the next layer", output_size)
                    self.Layers.append(nn.AdaptiveAvgPool2d((output_size,output_size)))
            
                if gaussian_noise == 1:
                    std = 0.001     #moderate disturbance
                    std_decay_rate = 0
                    self.Layers.append(  GaussianNoise(std, std_decay_rate) )

                kernel, stride, pool = layer_multiplicator * float(parallel_layer)
                kernel, stride, pool = int(kernel), int(stride), int(pool)
                #for the moment square but later can be implemented in x-y manner 
                Kx, Ky = kernel, kernel
                Sx, Sy = stride, stride
                Px,Py = 0,0     #no padding will be needed
                if batchnorm_switch == 1:
                    #if batchnorm is applied, then bias calc of conv layers is not needed -> performance gain
                    self.Layers.append(nn.ConvTranspose2d(IN, OUT , kernel_size=(Kx, Ky), stride=(Sx, Sy), output_padding=(Px, Py), bias = False )) #,output_padding = ()     #Attention ENhancer IN OUT Switched
                else:
                    self.Layers.append(nn.ConvTranspose2d(IN, OUT , kernel_size=(Kx, Ky), stride=(Sx, Sy), output_padding=(Px, Py) )) #,output_padding = ()     #Attention ENhancer IN OUT Switched

                output_size = int(self.AggregateMagnification)
                print("output size for adaptive pool", output_size)
                self.Layers.append(nn.AdaptiveAvgPool2d((output_size,output_size)))

                #self.Layers.append(nn.AvgPool2d((pool, pool), stride=(pool, pool)))         #old version of pooling caused blocky looking pics
                #cause MaxPool(Relu(x)) = Relu(MaxPool(x)) the activation is applied after the pooling, cause less parameters have to be activated -> proof of perf. inc pending
                self.Layers.append(nn.LeakyReLU(inplace = True))

                if dropout[0] == 1:
                    #if dropout switch is 1, then add dropout to layers
                    p = dropout[1] #with percentage of ...
                    self.Layers.append(nn.Dropout2d(p=p))

                if batchnorm_switch == 1:
                    self.Layers.append( nn.BatchNorm2d(OUT, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) )
            
        print("self.input_shape",self.input_shape)

        previous_Output_channels = OUT
        print("previous_Output_channels",previous_Output_channels)
        previous_parallel_layer_count = len(parallel_layers)
        print("previous_parallel_layer_count",previous_parallel_layer_count)
        IN = previous_Output_channels * previous_parallel_layer_count

        print("Discriminator conv pass through workaround")
        # 1x1 conv layer channel reduction
        self.Layers.append(nn.Conv2d(IN,1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) )      #1x1 channel reduction layer
        #To bring down the HxW to 1 for validity
        self.Layers.append(nn.AdaptiveAvgPool2d((1,1)))
        #Sigmoid activation for value range = 0...1
        self.Layers.append( nn.Sigmoid() )

        print("DISCRIMINATOR STRUCTURE")
        print(self.Layers)
        print("--------------INITIALIZING Discriminator DONE----------------")


    def forward(self, z):       
        z = z.view(self.opt.batch_size, 1, self.opt.latent_dim, self.opt.latent_dim_x)

        layerindex = 0

        for i in range(self.LayerCount):        #iterate forwards 
        #for i, layer in enumerate(self.Layers):        #iterate forwards# cause the layers were already added backwards?!!?!
            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = self.LayerDescription[i]
            
            #input("Something found??? doesnt have to... press any key to foreward next layer")
            #if layermag == 

            #print("layerindex",i, 'Layermagnification', layermagn)
            
            #IN,OUT, Kx, Ky,Sx,Sy,Px,Py,BN = self.LayerDescription[i]
            #layermagn, parallel_layer_list , channellist, batchnorm_switch = self.LayerDescription[i]
            #print("Layer",i," with Parameters", OUT,IN, Ky,Sx,Sy,Px,Py,BN)
            #IN, OUT = channellist
            #layer_multiplicator =  np.array(Sample_layerdict[layermagn])   #multiplicator for each layer, so that with every chosen kernelsize the stride and pooling is scaled accordingly

            data_branch = [None] * len(parallel_layers)      # cause single datapackage floods into the neurons, but cause of inception, multiple outputs 
            
            passed_data = z  #reshape data into  nd tensor

            for  index, parallel_layer in enumerate(parallel_layers):


                #print("parallel layer index", index)
                if self.opt.residual == True:
                    residual = self.Layers[layerindex](z)
                    layerindex +=1 

                if gaussian_noise == 1:
                    #data_branch[index] = self.Layers[layerindex](data_branch[index])  #gaussian noise
                    data_branch[index] = self.Layers[layerindex](passed_data)
                    #print(self.Layers[layerindex])

                    layerindex +=1 
                    data_branch[index] = self.Layers[layerindex](data_branch[index])

                else:
                    data_branch[index] = self.Layers[layerindex](passed_data)      #convlayer

                #print(self.Layers[layerindex])
                layerindex +=1

                data_branch[index] = self.Layers[layerindex](data_branch[index])   # pool layer
                #print(self.Layers[layerindex])
                layerindex +=1
                data_branch[index] = self.Layers[layerindex](data_branch[index])   # activation layer
                #print(self.Layers[layerindex])
                layerindex+=1

                if dropout[0] == 1:
                    #p = dropout[1]
                    data_branch[index] = self.Layers[layerindex](data_branch[index]) #dropout layer
                    #print(self.Layers[layerindex])
                    layerindex +=1


                if batchnorm_switch == 1:
                    data_branch[index] = self.Layers[layerindex](data_branch[index])
                    #print(self.Layers[layerindex])
                    layerindex +=1


                #if its the first element of the paralelle layers just inherent x and after that concatenate the channels, cause dimensions are the same in x,y batchsize
                if index == 0:

                    z = data_branch[index]
                    #print("x dtype", x.dtype)
                    #print("x.shape for intercept layer",x.shape)
                    #time.sleep(1)
                    #------------------------------------------------------
                else:
                    
                    z = torch.cat((z,data_branch[index]),1)
                #print(x.shape)

            
                
                if self.opt.residual == True:
                    residual_channellenght = int(residual.shape[1])
                    output_channellength = int(z.shape[1])
                    # if channels in layer before are more than now, just trash the last channels from past layer
                    if residual_channellenght > output_channellength:
                        residual_channellenght = output_channellength

                    passed_residual = torch.add(residual[:,:residual_channellenght,:,:], z[:,:residual_channellenght,:,:] )
                    passed_residual = torch.div(passed_residual,2)
                    #print("passed_residual",passed_residual)
                    #print("  passed_residual.shape", passed_residual.shape)
                    #print("z[:,2:,:,:].shape", z[:,2:,:,:].shape)
                    z = torch.cat(( passed_residual  , z[:,residual_channellenght:,:,:]),1)
                

        #Cause layercount just takes the inception layers the i has to be increased 3 more times to add 
        # to add 1x1 conv, adaptiv-avg-pool and sigmoid function
        i= -3
        #print("#################################################################")
        #print("adding conv1x1 layer")
        #print("Layerindex",i)
        #print("z.shape", z.shape)
        z = self.Layers[i](z)
        #print("z.shape", z.shape)
        i+=1
        #print("adding ADAPTIVE AVERAGE POOLING layer")
        #print("Layerindex",i)
        #print("z.shape", z.shape)
        z = self.Layers[i](z)
        #print("z.shape", z.shape)
        i+=1
        #print("adding SIGMOID layer")
        #print("Layerindex",i)
        #print("z.shape", z.shape)
        z = self.Layers[i](z)
        #print("z.shape", z.shape)

        validity = z
        validity = validity.view(self.opt.batch_size, 1)
        #print("discriminator validity shape", validity.shape)


        return validity




def save_image(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    #from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    #im.show()
    im.save(fp, format=format)


# Boolean for init all new child models
#first_time = True


def init_GAN(all_ness_params):
    global  Encoder_Netparameter_dict, Decoder_Netparameter_dict, opt, found


    opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace  =  all_ness_params 
    if opt.autoencoder == "off":
        pass
        #global Discriminator_Netparameter_dict
    ################################################################################################
    ###  EVOLUTIONARY/RANDOM NETWORK GENERATOR
    ################################################################################################

    NetGenParameters = {'img_size':opt.img_size[0],  'latent_dim':opt.latent_dim, 'Max_Lenght': opt.Max_Lenght, 'Max_parallel_layers': opt.Max_parallel_layers, 'opt':opt }
    netgen = Network_Generator(NetGenParameters)
    hyperopt_latent_dim = opt.latent_dim
    opt.latent_dim_x = int((opt.img_size[1]/ opt.img_size[0]) * opt.latent_dim)
    print("opt.latent_dim_x is  ",opt.latent_dim_x)

    #HYPERPARMS INIT BEGIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #opt.n_epochs =  HyperParameterspace['n_epochs']
    #opt.lr =  HyperParameterspace['lr']
    #opt.b1 =  HyperParameterspace['b1']
    #opt.b2 =  HyperParameterspace['b2']
    #opt.latent_dim =  HyperParameterspace['latent_dim']
    #print("opt.n_epochs" , HyperParameterspace['n_epochs'])
    #print("opt.lr", HyperParameterspace['lr'])
    #print("opt.b1",HyperParameterspace['b1'])Constant
    #print("opt.b2", HyperParameterspace['b2'])
    #print("opt.latent_dim", HyperParameterspace['latent_dim'])

    ########################################################################################
    #####################                   Inpainting Layers          #####################
    ########################################################################################  
    print("Generating Masking Layers")
    print("image size is", opt.img_size)
        
    opt.noisebool = True
    #std = 0.001     #moderate disturbance       #cant see anything at all
    opt.std = 0.01      #Hard Disurbance
    #opt.std = 0.001     #moderate disturbance
    #std = 0.0001    #light disturbance
    opt.std_decay_rate = 0 

    # if maskbool is None, Random masking is applied
    opt.maskbool = False
    #maskmean       x                       Y
    opt.maskmean = opt.img_size[1]/2 , opt.img_size[0]/2    #just the center for exploring  
    #maskmean = 50, 33
    #               x. Y                    
    opt.maskdimension = 50,25

    opt.LetterboxBool = False
    opt.LetterboxHeight = 30

    opt.PillarboxBool = False
    opt.PillarboxWidth = 10
    
    InpaintingParameters = {
        'opt': opt,
        'superresolution': (opt.superres, 2),
        'noise': (opt.noisebool, opt.std, opt.std_decay_rate),
        'mask': (opt.maskbool, opt.maskmean , opt.maskdimension),
        'Letterbox': (opt.LetterboxBool, opt.LetterboxHeight),
        'Pillarbox': (opt.PillarboxBool, opt.PillarboxWidth),

    }

    inpainting = Inpainting(InpaintingParameters)



    ########################################################################################
    #####################                   ENCODER                    #####################
    ########################################################################################    
    #ATTENTION: ENCODER MODEL JUST LIKE THE DECODER MODEL OUT OF
    ###               2D TRANSPOSE  Convolutions
    ###     cause the layermath just works better than going to conv layers, 
    # it's hard to get to a bigger picture with convs and pools

    #out/in Dimensionchg.    convolutional kernel-size & stride  Poolkernel
    #Sample_layerdict ={'1/8': [1,1,8], '1/4':[1,1,4], '1/2':[1,1,2], '1':[1,1,1], '1/2k':[2,2,1], '1/4k':[4,4,1], '1/8k':[8,8,1]}
    #Sample_layerdict ={'0.125': [1,1,8], '0.25':[1,1,4], '0.5':[1,1,2], '1':[1,1,1], '2':[2,2,1], '4':[4,4,1], '8':[8,8,1]}
    #Sample_layerdict ={'0.0078125': [1,1,128] ,'0.015625': [1,1,64] ,'0.03125': [1,1,32] ,'0.0625': [1,1,16] , '0.125': [1,1,8], '0.25':[1,1,4], '0.5':[1,1,2], '1.0':[1,1,1], '2.0':[2,2,1], '4.0':[4,4,1], '8.0':[8,8,1], '16.0':[16,16,1],'32.0':[32,32,1],'64.0':[64,64,1],'128.0':[128,128,1],'256.0':[256,256,1],}
    #TODO: GENERATE dict by function cause 0.03125=1/32


    # discription =         dim_ratio, paralell_layers(ker,stri,pool), channels, batchnorm
    # ATTENTION: FIRST ELEMENT OF PARALLEL LAYERS recieves the intercept from sbc


    ########################################################################################    
    #       EVOLUTIONAL GENERATION OF ENCODER NETWORK
    ########################################################################################    
    #opt.Population_Init_Threshold = 200
    #opt.Generation_limit = 100  #after x Child Models new mating begins to crossover the last generation with the recent trained ones.
    #opt.Generation_limit =  10 #opt.Population_Init_Threshold *2
    print("Trials at "+  str(opt.CurrentTrial)+ " and chosen Popuation Init threshold is," +str(opt.Population_Init_Threshold))
    #First time the Population init threshold is reached, mating will be initialized, so every model & Fitness is extracted from the previously calced models
    print(f"opt.current trial is {opt.CurrentTrial}")
    if opt.CurrentTrial > opt.Population_Init_Threshold and opt.first_time == True:
        print("INIT MATING ENCODER NOW")
        time.sleep(1)
        netgen.init_mating(opt)
        Encoder_Netparameter_dict = netgen.generate_children_from_parents("encoder",opt.Generation_limit,opt)
        opt.Enc_Child_index = 0
        #first_time = False juist after discriminator
        #input("CHECK IF ENCODER ARE INITATED")
    
    elif opt.CurrentTrial <= opt.Population_Init_Threshold:     #Population Init
        Valid_mag_train = netgen.init_all_magnification_trains('encoder')
        #print("All valid Valid_mag_train combos are")
        #print(Valid_mag_train)
        Valid_parallel_layer_train = netgen.init_all_parallel_layers()
        #print("All valid Valid_parallel_layer_train combos are",Valid_parallel_layer_train)
        LayerDescription = netgen.generate_random_Net("encoder",opt.No_latent_spaces, Valid_mag_train,Valid_parallel_layer_train )

    if opt.CurrentTrial > opt.Population_Init_Threshold:   # Crossover
        chosen_latent_spaces_ori = ['2', '4', '8', '16', '32', '64', '128']
        # make sure, that latent spaces are always tinier than input, else there is no compression 
        latent_dimensions = [x for x in chosen_latent_spaces_ori if int(x) < opt.img_size[0]]

        for latent_dimension in latent_dimensions:
            print(f"For {latent_dimension} there are,{len(Encoder_Netparameter_dict[latent_dimension])}  children net entrys")
        
        if opt.first_time_choosing_latent_size == True:
            while True:         
                try:
                    opt.latent_dim = int(input(f"Choose latent dimension from {latent_dimensions} or type 1 for continuing random search"))
                    if opt.latent_dim == 1:
                        print(f"overwrite latent dim with hyperopt guess")
                        opt.latent_dim = hyperopt_latent_dim
                        opt.latent_dim_x = int((opt.img_size[1]/ opt.img_size[0]) * opt.latent_dim)
                        print(f"Continue with random network search")
                        opt.Population_Init_Threshold = opt.CurrentTrial + opt.Generation_limit
                        
                        Valid_mag_train = netgen.init_all_magnification_trains('encoder')
                        Valid_parallel_layer_train = netgen.init_all_parallel_layers()
                        LayerDescription = netgen.generate_random_Net("encoder",opt.No_latent_spaces, Valid_mag_train,Valid_parallel_layer_train )
                        
                    else:
                        opt.latent_dim_x = int((opt.img_size[1]/ opt.img_size[0]) * opt.latent_dim)
                        opt.first_time_choosing_latent_size = False
                        found = False
                    break
                except:
                    PrintException()
                    continue

        
        while found == False:    
            try:
                print(f"Try to get a child arch with for ENCODER with latent dim of{opt.latent_dim} and Index {opt.Enc_Child_index} with {len(Encoder_Netparameter_dict[str(opt.latent_dim)])}")
                LayerDescription = netgen.get_child_arch("encoder",str(opt.latent_dim), str(opt.Enc_Child_index),Encoder_Netparameter_dict)
                print("Entry found. continue with NET INIT")
                opt.chosen_latent_dim = int(opt.latent_dim)
                print(f"Chosen latent dimension is {opt.chosen_latent_dim}")
                #break
                found = True
                break
            except:
                PrintException()
                opt.Enc_Child_index += 1
                if opt.Enc_Child_index >= len(Encoder_Netparameter_dict[latent_dimension]):
                    print("End of Encoder Netparameter dict reached exit")
                    break
                continue
        


    
    
    print("ENCODER NETWORK LAYERDISCRIPTION")
    print(LayerDescription) 


    '''
    #               gaussian Noise,         magnification           paralell layers   channels    Dropout                  Batch norm
    layerlist = [ random.randint(0, 1 ), str(chosen_mag_train[i]), parallel_layer ,  [IN,OUT], random.randint(0, 1 )   , random.randint(0, 1) ]

    LayerDescription = [ 
                                [0, '1'   , [1]      , [1,4]  ,1  , 0],
                                [0, '0.5' , [1,2]    , [4,8]  ,1  , 1 ],
                                [0, '0.5' , [1,2]    , [8,10] ,1  , 1 ],
                                [0, '0.25', [1,2,4,8], [10,5] ,1  , 1],

                            ] 
    '''
    #                            ['1', [1,2,4,8,16] , [32,1] , 0 ],

    #magnification = np.divide(opt.latent_dim ,opt.img_size[0])
    try:
        start_dim = opt.chosen_latent_dim #evogen
        print(f"Overwrote latent dimension with"[opt.chosen_latent_dim])
    except:
        start_dim = opt.latent_dim #randomgen

    end_dim = opt.img_size[0]   

    magnification = end_dim / start_dim  #(32*32)/(128*128) = 16 -> versechzehnfachung des bildes


    input_shape = (opt.batch_size,1, opt.img_size[0],opt.img_size[1])
    NetParameters = {'LayerDescription': LayerDescription, 'input_shape': input_shape, 'SpacialBoxcounting':BoxcountEncoder, 'magnification':magnification, 'opt':opt, 'device': device}

    #Network_Parameter_list = [NetParameters]
    #Init Encoder
    encoder = Encoder(NetParameters)
    count_parameters(encoder)
    
    EncoderNetParameters = NetParameters
    #del EncoderNetParameters['opt']['Encoder_Netparameter_dict']
    #print(EncoderNetParameters)
    #input("JETZT ABER")
    ########################################################################################
    #####################               DECODER/GENERATOR              #####################
    ########################################################################################   


    # VALUE INITIALIZATION Decoder  MODEL
    # 2d conv transpose layerdictionary

    #out/in Dimensionchg.    transpose kernel-size & stride  Poolkernel
    #Sample_layerdict ={'1/8': [1,1,8], '1/4':[1,1,4], '1/2':[1,1,2], '1':[1,1,1], '2':[2,2,1], '4':[4,4,1], '8':[8,8,1]}

    try:
        start_dim = opt.chosen_latent_dim #evogen
    except:
        start_dim = opt.latent_dim #randomgen    end_dim = opt.img_size[0]   

    magnification = end_dim / start_dim  #(32*32)/(128*128) = 16 -> versechzehnfachung des bildes

    #NEW METHODE#################################################################################
    '''
    Each list of entrys discribes a inception layer
    with the out/in dimension ratio, list of paralell layers/filters, the corresponding in/output channels
    and if batchnormalization is performed or not, the last digit tells, 
    if the ouput of the incepiton layer is passing the residual  and not just the direct mapping (resnet) 

    '''
    # discription =         dim_ratio, paralell_layers, channels, batchnorm
    # ATTENTION: INPUT layer has to be with 1 input channel
    #            Last Layer has to be with 1 output channel and no parallel layers to match the output
    # ATTENTION: FIRST ELEMENT OF PARALLEL LAYERS recieves the intercept from sb


    #######
    #       EVOLUTIONAL GENERATION OF DECODER NETWORK
    ####
    #First time the Population init threshold is reached, mating will be initialized, so every model & Fitness is extracted from the previously calced models
    if opt.CurrentTrial > opt.Population_Init_Threshold and opt.first_time == True:
        print("INIT MATING DECODER NETORKS NOW")
        #netgen.init_mating(opt)
        Decoder_Netparameter_dict = netgen.generate_children_from_parents("decoder",opt.Generation_limit,opt)
        opt.Dec_Child_index = 0
        if opt.autoencoder == "on":
            opt.first_time = False
        #input("CHECK IF DECODER ARE INITATED")

    elif opt.CurrentTrial <= opt.Population_Init_Threshold:     #Population Init
        Valid_mag_train = netgen.init_all_magnification_trains('decoder')
        #print("All valid Valid_mag_train combos are")
        #print(Valid_mag_train)
        Valid_parallel_layer_train = netgen.init_all_parallel_layers()
        #print("All valid Valid_parallel_layer_train combos are",Valid_parallel_layer_train)
        LayerDescription = netgen.generate_random_Net("decoder",opt.No_latent_spaces, Valid_mag_train,Valid_parallel_layer_train )
        #if opt.autoencoder == "on":

    if opt.CurrentTrial > opt.Population_Init_Threshold:   # Crossover
        '''
        #Cause latent dim should be the same when generating child archs, if not, then models arnt compatible
        for latent_dimension in latent_dimensions:
        '''
        #for trys in len(Decoder_Netparameter_dict[opt.chosen_latent_dim]):   
        print(f"No Entries in decoder netparameter dict with latent dim of {opt.chosen_latent_dim}  :  {len(Decoder_Netparameter_dict[str(opt.chosen_latent_dim)])}")
        print(Decoder_Netparameter_dict[opt.chosen_latent_dim])
        while True:
            try:
                print(f"Try to get ChildArch from Generated Dict with latent_dim={opt.chosen_latent_dim} and model index {opt.Dec_Child_index} with Layerdiscription:")
                LayerDescription = netgen.get_child_arch("decoder",str(opt.chosen_latent_dim), str(opt.Dec_Child_index), Decoder_Netparameter_dict)
                print("Entry found. continue with NET INIT")
                break
            except:
                opt.Dec_Child_index += 1
                print("THIS IS THE WHOLE DECODER NETPARAMETER DICT")
                print(Decoder_Netparameter_dict[str(opt.chosen_latent_dim)])
                input(f"no model found with searched key, press key to continue")
                PrintException()
                if opt.Dec_Child_index >= opt.Generation_limit:
                    print("no model found")
                    opt.first_time = True
                    opt.Population_Init_Threshold += opt.Generation_limit
                    opt.first_time_choosing_latent_size = True      
                    break
        
    '''
    if opt.CurrentTrial >= opt.Population_Init_Threshold and opt.Dec_Child_index >= opt.Generation_limit:   #new generation 
        print("REINIT MATING for new generation")
        netgen.init_mating(opt)
        Decoder_Netparameter_dict = netgen.generate_children_from_parents("decoder",opt.Generation_limit,opt)
        opt.Dec_Child_index = 0
        while True:
            try:
                LayerDescription = netgen.get_child_arch("decoder",opt.chosen_latent_dim, opt.Dec_Child_index, Decoder_Netparameter_dict)
                print("Entry found. continue with NET INIT")
                break
            except:
                PrintException()
                opt.Dec_Child_index += 1
                if opt.Dec_Child_index >= opt.Generation_limit:
                    print("no model found")                    
                    break
    '''



    '''
    ###############
    #       GENERATING deCODER NETWORK
    ###############

    Valid_mag_train = netgen.init_all_magnification_trains('decoder')
    #print("All valid Valid_mag_train combos are")
    #print(Valid_mag_train)

    Valid_parallel_layer_train = netgen.init_all_parallel_layers()
    #print("All valid Valid_parallel_layer_train combos are",Valid_parallel_layer_train)

    LayerDescription = netgen.generate_random_Net("decoder",opt.No_latent_spaces, Valid_mag_train,Valid_parallel_layer_train )
    '''
    print("DECODER NETWORK LAYERDISCRIPTION")
    print(LayerDescription) 


    input_shape = (opt.batch_size,1, opt.img_size[0],opt.img_size[1])

    NetParameters = {'LayerDescription': LayerDescription, 'input_shape': input_shape,'magnification': magnification, 'opt':opt, 'No_latent_spaces': opt.No_latent_spaces}

    # Initialize Decoder
    decoder = Decoder(NetParameters)
    count_parameters(decoder)
    DecoderNetParameters = NetParameters





    ########################################################################################
    #####################            DISCRIMINATOR                  #####################
    ########################################################################################   




    if opt.autoencoder == "off":
        #######
        #       EVOLUTIONAL GENERATION OF DISCRIMINATOR NETWORK
        ####
        #First time the Population init threshold is reached, mating will be initialized, so every model & Fitness is extracted from the previously calced models
        if opt.CurrentTrial > opt.Population_Init_Threshold and opt.first_time == True:
            print("INIT MATING NOW")
            #netgen.init_mating(opt)
            Discriminator_Netparameter_dict = netgen.generate_children_from_parents("discriminator",opt.Generation_limit,opt)
            opt.Dis_Child_index = 0
            opt.first_time = False
            #opt.Population_Init_Threshold += opt.Generation_limit
        elif opt.CurrentTrial <= opt.Population_Init_Threshold:     #Population Init
            Valid_mag_train = netgen.init_all_magnification_trains('discriminator')
            #print("All valid Valid_mag_train combos are")
            #print(Valid_mag_train)
            Valid_parallel_layer_train = netgen.init_all_parallel_layers()
            #print("All valid Valid_parallel_layer_train combos are",Valid_parallel_layer_train)
            LayerDescription = netgen.generate_random_Net("discriminator",opt.No_latent_spaces, Valid_mag_train,Valid_parallel_layer_train )

        if opt.CurrentTrial > opt.Population_Init_Threshold:   # Crossover
            #for trys in len(Discriminator_Netparameter_dict[opt.chosen_latent_dim]):
            while True:
                try:
                    LayerDescription = netgen.get_child_arch("discriminator",opt.chosen_latent_dim, str(opt.Dis_Child_index),Discriminator_Netparameter_dict)
                    print("Entry found. continue with NET INIT")
                    break
                except:
                    PrintException()
                    opt.Dis_Child_index += 1
                    if opt.Dis_Child_index >= opt.Generation_limit:
                        print("no model found")
                        opt.first_time = True
                        opt.Population_Init_Threshold += opt.Generation_limit
                        opt.first_time_choosing_latent_size = True      

                        break
        '''
        if opt.CurrentTrial >= opt.Population_Init_Threshold and opt.Dis_Child_index >= opt.Generation_limit:   #new generation 
            print("REINIT MATING for new generation")
            netgen.init_mating(opt)
            Discriminator_Netparameter_dict = netgen.generate_children_from_parents("discriminator",opt.Generation_limit,opt)
            opt.Dis_Child_index = 0
            while True:
                try:
                    LayerDescription = netgen.get_child_arch("discriminator",opt.chosen_latent_dim, opt.Dis_Child_index,Discriminator_Netparameter_dict)
                    print("Entry found. continue with NET INIT")
                    break
                except:
                    PrintException()
                    opt.Dis_Child_index += 1
                    if opt.Dis_Child_index >= opt.Generation_limit:
                        print("no model found")
                        break
            #LayerDescription = netgen.get_child_arch("discriminator",latent_dimension, opt.Dis_Child_index,Discriminator_Netparameter_dict)
        '''
        print("DISCRIMINATOR CHILD LAYERDISCRIPTION", LayerDescription)



        input_shape = (opt.batch_size, opt.latent_dim**2)

        NetParameters = {'LayerDescription': LayerDescription, 'input_shape': input_shape, 'No_latent_spaces':opt.No_latent_spaces, 'opt':opt,  'device': device}

        #INIT DIS NETWORK
        discriminator = Discriminator(NetParameters)
        count_parameters(encoder)
        DiscriminatorNetParameters = NetParameters




    else:
        discriminator = None
        DiscriminatorNetParameters = None

    '''
    dict_length = len(Encoder_Netparameter_dict[str(opt.latent_dim)])
    if dict_length > 482:
        input(f"Encoderdict lenght is altered with len {dict_length}")
    '''
    #HYPERPARMS INIT ENDy<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return opt, inpainting, InpaintingParameters, encoder,EncoderNetParameters,  decoder, DecoderNetParameters, discriminator, DiscriminatorNetParameters




# -------------------------------------------------------------------------------------
#  Training
# -------------------------------------------------------------------------------------
def TrainGAN(opt,trainDataloader, inpainting ,  encoder, decoder, discriminator , BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss):
    #global all_ness_params
    ########################################################################################
    #####################               LOSSES                         #####################
    if opt.autoencoder == "off":
        adversarial_loss = torch.nn.BCELoss() # Use binary cross-entropy loss
    pixelwise_loss = torch.nn.L1Loss()

    ########################################################################################
    #####################               Send to device                 #####################   

    if device == "cuda":
        try:
            BoxcountEncoder.cuda()
        except:
            pass
        try:
            inpainting.cuda()
        except:
            PrintException()
            pass
        encoder.cuda()
        decoder.cuda()
        if opt.autoencoder == "off":
            discriminator.cuda()
            adversarial_loss.cuda()
        pixelwise_loss.cuda()

    ########################################################################################
    #####################               Optimizers                     #####################
    optimizer_AE = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    if opt.autoencoder == "off":
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_G = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor

    Loss_Now = 100000.0 #Starting loss just to ensure, that first model is best after 1st training
    Dis_Loss = 100000.0
    TrailingLoss = 10.0
    #LossLastRound = 100.00
    running_mean_list = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    opt.max_happens = 500
    Happend_counter = 1
    config_counter = 0
    opt.breaker = False
    import config
    dataloaderlenght = len(trainDataloader)

    printstatement = False


    for epoch in range(opt.n_epochs):
        printcounter = 0
        for i, BatchToBePredicted in enumerate(trainDataloader):
            start = time.perf_counter()
            display = False
            config_counter += 1
            if config_counter == 50:
                config = reload(config) #to alter the configs on the fly while training/testing and importing here to always import changes made to the config.py file
                config_file = config.config_file()
                opt = config_file.set_opt_parameters(config_file.ON_OFF_Switch,opt)
                config_counter = 0
                printstatement = True

            if opt.breaker:
                #if user specifys opt.breaker = True in config.py, 
                break

            #MODE 1: PREDICT FROM CNN BC
            if opt.Mode == 1:
                imgs, __, __, __, __ = BatchToBePredicted
                BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16 = BoxCountEnc_model.predict( BoxcountEncoder, device,BatchToBePredicted, display)
                #BatchToBePredicted = [imgs, SBC2,SBC4,SBC8,SBC16]
                #NumpyencImg2, NumpyencImg4, NumpyencImg8, NumpyencImg16 = BoxCountEnc_model.predict(Modelname,BoxcountEncoder ,device,imgs)
                #SBC = [BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16]
                #SBC = SBC.float()

            #MODE 2: USE LABELS FROM CPU BOXCOUNT

            if opt.Mode == 2:
                imgs, BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16 = BatchToBePredicted
                BCR_LAK_map_2.float(), BCR_LAK_map_4.float(), BCR_LAK_map_8.float(), BCR_LAK_map_16.float()

            real_labels_2 = Variable(BCR_LAK_map_2.type(Tensor))
            real_labels_2.to(device)

            real_labels_4 = Variable(BCR_LAK_map_4.type(Tensor))
            real_labels_4.to(device)

            real_labels_8 = Variable(BCR_LAK_map_8.type(Tensor))
            real_labels_8.to(device)

            real_labels_16 = Variable(BCR_LAK_map_16.type(Tensor))
            real_labels_16.to(device)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            real_imgs.to(device)

            #Data on which the pixelloss is calced - because real_imgs are altered by inpainting module
            ori_imgs = Variable(imgs.type(Tensor))
            ori_imgs.to(device)

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            valid.to(device)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            fake.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_AE.zero_grad()

            #if you hit the limit of updatemaskevery, when random mask mode is true. choose new random mask
            if printcounter == opt.UpdateMaskEvery:
                if inpainting.randdommask == True:
                    # -----------------
                    #  UPDATE INPAINTING
                    # -----------------
                    #opt.maskbool = False #This should be enough for random init of masking/inpainting
                    #should be not be neccecary
                    opt.maskmean = (torch.randint(50,150,(1,),device=torch.device(opt.device))[0]  , torch.randint(50,150,(1,),device=torch.device(opt.device))[0]  )
                    opt.maskdimension = (torch.randint(1,50,(1,),device=torch.device(opt.device))[0]  , torch.randint(1,50,(1,),device=torch.device(opt.device))[0]  )
                    print("Reinit inpainting with  " ,opt.maskmean, opt.maskdimension)
                    #ReInitilization of Inpainting layer
                    InpaintingParameters = {
                        'opt': opt,
                        'superresolution': (opt.superres, 2),
                        'noise': (opt.noisebool, opt.std, opt.std_decay_rate),
                        'mask': (opt.maskbool, opt.maskmean , opt.maskdimension),
                        'Letterbox': (opt.LetterboxBool, opt.LetterboxHeight),
                        'Pillarbox': (opt.PillarboxBool, opt.PillarboxWidth),
                    }
                    inpainting = Inpainting(InpaintingParameters)
                    printcounter = 0 #Reset Printcounter

            batches_done = epoch * dataloaderlenght + i

            if batches_done % opt.sample_interval == 0:
                input_imgs = real_imgs[:,0,:,:].detach().cpu().numpy() 

            real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16 = inpainting(real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16)
            printcounter +=1
            
            start_autoencoder = time.perf_counter()
            encoded_imgs = encoder(real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16)
            end_encoder = time.perf_counter()

            decoded_imgs = decoder(encoded_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16 )
            end_autoencoder = time.perf_counter()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if opt.autoencoder == "off":
                start_discriminator = time.perf_counter()
                
                predicted = discriminator(encoded_imgs)
                #Sometime cuda asynchronus call fail happens, so filtering out nans and infs and setting them to 0 
                filtered_predicted = predicted.detach()
                filtered_predicted[torch.isnan(filtered_predicted)] = 0
                filtered_predicted[filtered_predicted == float("Inf")  ] = 0

                # Loss measures generator's ability to fool the discriminator
                discr_loss = 0.001 * adversarial_loss(filtered_predicted, valid) 
                Pixelloss =  0.999 * pixelwise_loss(decoded_imgs, ori_imgs)  #compare against ori_imgs, because real_imgs were altered by iz
                AE_loss = discr_loss + Pixelloss
                AE_loss[torch.isnan(AE_loss)] = 1.0
                AE_loss.backward()
                AutoEncoder_loss = float(AE_loss.item())
                optimizer_AE.step()

                encoder.eval()  # to disable dropout and fix the encoder network
                optimizer_D.zero_grad()

                # Sample noise as fake latent variable
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim*opt.latent_dim_x ))))
                z_fake = encoder(real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16)
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(z), valid)
                fake_loss = adversarial_loss(discriminator(z_fake), fake)

                #Sometime cuda asynchronus call fail happens, so filtering out nans and infs and setting them to 0 
                filtered_real_loss = real_loss  #.detach()
                filtered_real_loss[torch.isnan(filtered_real_loss)] = 0
                filtered_real_loss[filtered_real_loss == float("Inf")  ] = 0
 
                filtered_fake_loss = fake_loss   #.detach()
                filtered_fake_loss[torch.isnan(filtered_fake_loss)] = 0
                filtered_fake_loss[filtered_fake_loss == float("Inf")  ] = 0


                d_loss = 0.5 * (filtered_real_loss + filtered_fake_loss)
                d_loss.backward()
                optimizer_D.step()

                encoder.train()
                optimizer_G.zero_grad()
                #New fake latent space with backprop of generator(encoder) to match the gaussian distribution
                z_fake = encoder(real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16)
                g_loss = adversarial_loss(discriminator(z_fake),valid)
                g_loss.backward()
                optimizer_G.step()
                gen_loss = float(g_loss.item())

                Dis_Loss = float(d_loss.item())


                if Dis_Loss == 50.0:
                    print("d_loss is 50 and AE_loss should be nan...sleep 2 and break")
                    time.sleep(2)
                    Loss_Now = 100000.0 -float(i)
                    break
                else:
                    Loss_Now = float(AE_loss.item()) + float(d_loss.item())

                
                LossThisRound = Dis_Loss + AutoEncoder_loss
                '''
                #if round(LossThisRound,5) == round(LossLastRound,5):
                #    Happend_counter +=1
                #    if Happend_counter >=opt.max_happens:
                #        #if this happend 10 times, then break
                #        print("Non Decreasing/Constant Discriminator Loss happend ", Happend_counter)
                #        print("Last trailing Loss: "+ str(TrailingLoss)+ " Constant Discriminator loss happend "+str(Happend_counter)+"/"+str(opt.max_happens))
                #        break
                #    else:
                #        #print("Loss Converging")
                #        pass
                '''
            
                if printstatement:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [AE loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, dataloaderlenght, Dis_Loss, AutoEncoder_loss,gen_loss)
                    )
                    print(f"Last trailing Loss: {TrailingLoss}")
                    running_mean = round(np.mean(running_mean_list),6)
                    print(f"Running Mean Loss: {running_mean}")


            else:

                AE_loss = pixelwise_loss(decoded_imgs, ori_imgs)  #compare against ori_imgs, because real_imgs were altered by inpainting class
                AE_loss[torch.isnan(AE_loss)] = 1.0
                AE_loss.backward()
                optimizer_AE.step()
                AutoEncoder_loss = float(AE_loss.item())                
                #AutoEncoder_loss = float(AE_loss.item())
                LossThisRound = AutoEncoder_loss
                
                if printstatement:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d]  [AE loss: %f]"
                        % (epoch, opt.n_epochs, i, dataloaderlenght, AutoEncoder_loss))
                    print(f"Last trailing Loss: {TrailingLoss}")
                    running_mean = round(np.mean(running_mean_list),6)
                    print(f"Running Mean Loss: {running_mean}")

            running_mean_list = np.append(running_mean_list, LossThisRound) # append value to running mean
            running_mean_list = np.delete(running_mean_list,0)    #delete first element
            


            #print(running_mean_list)
            '''
            if batches_done % opt.sample_interval == 0:
            print("trying to sample")
            #input()
            #SBC = [real_labels_2, real_labels_4, real_labels_8, real_labels_16]
            #input_imgs = real_imgs[:,0,:,:].detach().cpu().numpy() #batch, 1, y,x
            #inpainted_imgs = real_imgs[:,0,:,:].detach().cpu().numpy()       
            #gen_imgs = decoded_imgs[:,0,:,:].detach().cpu().numpy()

            #sample_image(n_row=10, batches_done=batches_done, opt=opt, input_imgs = input_imgs, inpainted_imgs=inpainted_imgs, gen_imgs=gen_imgs)
            #input("HAAAALT")
            '''
                        
            if opt.CurrentTrial <= opt.Population_Init_Threshold:
                TrailingLoss = np.multiply(TrailingLoss, 0.98)
            else:    
                #scaling = 0.5
                scaling = 0.12
                #scaling = 0.025
                multiplicator = 1.0 - (AutoEncoder_loss * scaling)
                TrailingLoss = np.multiply(TrailingLoss, multiplicator)  
        
            end = time.perf_counter()
            
            if printstatement:
                print(f"Lapse time:{round(end-start,6)}s   Encoder time: {round(end_encoder - start_autoencoder,6)}s    Decoder time: {round(end_autoencoder - end_encoder,6)} ")
                #resetting printstatement
                printstatement = False

            #LossLastRound = LossThisRound
            if LossThisRound >= TrailingLoss:
                #If The loss this round is Higher than the mean of the trailing loss, then break training, cause model isn't going anywhere 
                print("Breaking, cause model doesnt converge anymore, but please check anyway")
                time.sleep(1)
                break

    


    Loss_Now = running_mean


    print("Best loss so far  :", previous_Best_Loss)
    print("loss of this model:", running_mean)

    if previous_Best_Loss == None:
        previous_Best_Loss = Loss_Now
    elif previous_Best_Loss >  Loss_Now:
        previous_Best_Loss = Loss_Now




    print("Best loss so far  :", previous_Best_Loss)
    print("loss of this model:", running_mean)

    return Loss_Now, running_mean, previous_Best_Loss, Dis_Loss, encoder, decoder, discriminator


def TrainGAN_with(all_ness_params):
    global opt
    opt, Dataset, trainDataloader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace  =  all_ness_params 

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<HYPERPARAMETERS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    opt.n_epochs =  HyperParameterspace['n_epochs']
    opt.lr =  HyperParameterspace['lr']
    opt.b1 =  HyperParameterspace['b1']
    opt.b2 =  HyperParameterspace['b2']

    if opt.CurrentTrial <= opt.Population_Init_Threshold :
        print("Altering latent dim through hyperopt. Should only happen, when in pop init phase")
        opt.latent_dim =  HyperParameterspace['latent_dim']
    
    opt.latent_dim_x = int((opt.img_size[1]/ opt.img_size[0]) * opt.latent_dim)
    
    print("opt.lr", opt.lr)
    print("opt.b1",opt.b1)
    print("opt.b2", opt.b2)
    print("opt.latent_dim", opt.latent_dim)
    print("opt.latent_dim_x is  ",opt.latent_dim_x)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<HYPERPARAMETERS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    Not_Working = 0
    
    opt, inpainting, InpaintingParameters ,encoder,EncoderNetParameters,  decoder, DecoderNetParameters, discriminator, DiscriminatorNetParameters  = init_GAN(all_ness_params)
    
    while True:
        try:
            Loss_Now, AE_loss, previous_Best_Loss,Dis_Loss,  encoder, decoder, discriminator = TrainGAN(opt,trainDataloader,inpainting , encoder, decoder, discriminator , BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss)
            break
        except:
            PrintException()
            print("Model could not be initialized, sleep 1s and try again with next child arch")
            print("Child_index "+ str(opt.Enc_Child_index ))
            opt.Enc_Child_index += 1
            opt.Dec_Child_index += 1
            opt.Dis_Child_index += 1
            try:    
                print("Clear up memory by deleting the models from vram")
                del encoder
                del decoder
                if opt.autoencoder == "off":
                    del discriminator
                del inpainting
            except:
                pass

            torch.cuda.empty_cache()

            opt, inpainting, InpaintingParameters ,encoder,EncoderNetParameters,  decoder, DecoderNetParameters, discriminator, DiscriminatorNetParameters  = init_GAN(all_ness_params)
            
            Not_Working += 1
            if Not_Working >= 1000:
                PrintException()
                raise Exception("INIT or Training not possible")
                break

            continue
        
    # save all models, if in pop init phase, but when in evolutionary search just save, when better model was found with LOWER LOSS 
    if Loss_Now <= previous_Best_Loss or opt.CurrentTrial < opt.Population_Init_Threshold or opt.SaveEveryModel == True :
        '''
        saveplace = FileParentPath
        saveplace +="/models/"
        saveplace +="/GAN/"
        saveplace += opt.ProjectName +"/"
        '''
        if sys.platform == "linux" or sys.platform == "darwin":
            saveplace = FileParentPath + "/models/GAN/" + opt.ProjectName + "/"
        elif sys.platform == "win32":
            saveplace = FileParentPath + "\\models\\GAN\\" + opt.ProjectName + "\\"


        saveplace += str(int(time.time())) + "_"        #to append something unique to filename preventing overwriting

        print(EncoderNetParameters)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(EncoderNetParameters['LayerDescription'])
        #print(len(EncoderNetParameters['LayerDescription']))

        #ENCODER########################################
        NetParametersSaveplace = saveplace+"Loss" + str(round(AE_loss,6)) +"---_ENCODER" +".netparams"
        with open(NetParametersSaveplace, "wb") as f:
            pickle.dump(EncoderNetParameters, f)
        
        stateDictSaveplace = saveplace+"Loss" + str(round(AE_loss,6)) +"---_ENCODER" +".model"
        torch.save(encoder.state_dict(), stateDictSaveplace)

        #DECODER########################################
        NetParametersSaveplace = saveplace+"Loss" + str(round(AE_loss,6)) +"---_DECODER" +".netparams"
        with open(NetParametersSaveplace, "wb") as f:
            pickle.dump(DecoderNetParameters, f)
        
        stateDictSaveplace = saveplace + "Loss" + str(round(AE_loss,6)) +"---_DECODER.model"
        torch.save(decoder.state_dict(), stateDictSaveplace)

        if opt.autoencoder == "off":
            #Discriminator#####################################
            NetParametersSaveplace =  saveplace+"Loss" + str(round(Dis_Loss,6)) +"---_DISCRIMINATOR" +".netparams"
            with open(NetParametersSaveplace, "wb") as f:
                pickle.dump(DiscriminatorNetParameters, f)
            
            stateDictSaveplace =   saveplace +"Loss" + str(round(Dis_Loss,6)) +"---_DISCRIMINATOR.model"
            torch.save(discriminator.state_dict(), stateDictSaveplace)

        print("Model Saved")
        previous_Best_Loss = Loss_Now

    else:
        print("Loss was higher/worse than previous best model")

    return {'loss': Loss_Now, 'status': STATUS_OK}



def begin_training(all_ness_params):
    global previous_Best_Loss
    #global all_ness_params
    opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace  =  all_ness_params 
    opt.Enc_Child_index = 0
    opt.Dec_Child_index = 0
    opt.Dis_Child_index = 0
    opt.first_time_choosing_latent_size = True

    #Source: https://github.com/hyperopt/hyperopt/issues/267
    #To save trials object to pick up where you left
    def run_trials(ProjectName,all_ness_params):
        opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace  =  all_ness_params 
        #ATTENTION: If you want to begin training anew, then you have to delete the .hyperopt file
        TrialsSaveplace = FileParentPath
        TrialsSaveplace +=  "/Hyperoptimization/"+ str(ProjectName) +".hyperopt" 
        trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
        max_trials = 5  # initial max_trials. put something small to not have to wait
        opt.max_trials = max_trials
        try:  # try to load an already saved trials object, and increase the max
            trials = pickle.load(open(TrialsSaveplace, "rb"))
            print("Found saved Trials! Loading...")
            max_trials = len(trials.trials) + trials_step
            opt.CurrentTrial = len(trials.trials)
            opt.max_trials = max_trials
            print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
            step = opt.Generation_limit
            print(f"pop init thresh is {opt.Population_Init_Threshold}, halting at {opt.Population_Init_Threshold+step}, while current trial is {opt.CurrentTrial}")
            if opt.Population_Init_Threshold + step+10 <= opt.CurrentTrial: input("STep reached. PLEASE DELLETEEE")
            opt.first_time = True # reset opt.first_time to generate new generation with maybe other latent space

        except:  # create a new trials object and start searching
            PrintException()
            trials = Trials()
            opt.CurrentTrial = 1


        all_ness_params =  opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace

        lowest_loss = fmin(TrainGAN_with, all_ness_params, algo=tpe.suggest, max_evals=max_trials, trials=trials)
        #print("Lowest achieved loss so far:", lowest_loss)
        
        #if last loss is better than prevous loss, than overwrite best previous loss
        if previous_Best_Loss == None or previous_Best_Loss > trials.losses()[-1]:
            previous_Best_Loss = trials.losses()[-1]
        #print(f"trials.losses(){trials.losses()[-1]}")
        # save the trials object
        with open(TrialsSaveplace, "wb") as f:
            pickle.dump(trials, f)

        all_ness_params =  opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace

        return all_ness_params

    # loop indefinitely and stop whenever you like by setting MaxTrys
    MaxTrys = 10000
    #initial_population = 50  #Initial population of 250 trys for every length and depth in network
    #Interrupt Trials
    äInterrupt_trials_index = 0
    #Both parallell layers and max lenght has to be 2 at least for permutations 
    import config

    config_file = config.config_file()
    ON_OFF_Switch = config_file.ON_OFF_Switch
    opt = config_file.set_opt_parameters(ON_OFF_Switch,opt)
    HyperParameterspace = config_file.set_Hyperparameterspace(config_file.ON_OFF_Switch_Hyperparams ,HyperParameterspace)
    all_ness_params = opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss , HyperParameterspace  

    for TotalTrials in range(MaxTrys):
        config = reload(config) #to alter the configs on the fly while training/testing and importing here to always import changes made to the config.py file
        config_file = config.config_file()
        ON_OFF_Switch = config_file.ON_OFF_Switch
        #print("Onn OF Switch is ",ON_OFF_Switch)
        opt = config_file.set_opt_parameters(ON_OFF_Switch,opt)
        HyperParameterspace = config_file.set_Hyperparameterspace(config_file.ON_OFF_Switch_Hyperparams ,HyperParameterspace)
        #all_ness_params = opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss , HyperParameterspace  
        all_ness_params =  run_trials( opt.ProjectName,all_ness_params)    


# -------------------------------------------------------------------------------------
#  Testing
# -------------------------------------------------------------------------------------

def TestGAN(all_ness_params):
    import config

    opt, Dataset, testDataloader, inpainting, InpaintingParameters ,encoder,EncoderNetParameters,  decoder, DecoderNetParameters,  discriminator , DiscriminatorNetParameters, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace = all_ness_params

    inpainting = Inpainting(InpaintingParameters) #initialize inpainting class

    ########################################################################################
    #####################               LOSSES                         #####################   

    if opt.autoencoder == "off":
        adversarial_loss = torch.nn.BCELoss()  # Use binary cross-entropy loss

    pixelwise_loss = torch.nn.L1Loss()   # mean squared error pixel-wise loss

    ########################################################################################
    #####################               Send to device                 #####################
    print(f"Chosen Testing device == {opt.device}")
    
    if opt.device == "cuda":
        try:
            BoxcountEncoder.cuda()
        except:
            pass
        try:
            inpainting.cuda()
        except:
            PrintException()
            pass

        encoder.cuda()
        decoder.cuda()
        if opt.autoencoder == "off":
            discriminator.cuda()
            adversarial_loss.cuda()
        pixelwise_loss.cuda()

    Tensor = torch.cuda.FloatTensor if opt.device == "cuda" else torch.FloatTensor
    config_counter = 0

    for i, BatchToBePredicted in enumerate(testDataloader):
        torch.no_grad() #no gradients required during testing.

        config = reload(config) #to alter the configs on the fly while training/testing and importing here to always import changes made to the config.py file
        config_file = config.config_file()
        opt = config_file.set_opt_parameters(config_file.ON_OFF_Switch,opt)

        inpainting = Inpainting(opt.InpaintingParameters)
        if opt.device == "cuda":
            inpainting.cuda() 


        #MODE 1: PREDICT FROM CNN BC
        if opt.Mode == 1:
            imgs, __, __, __, __ = BatchToBePredicted
            BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16 = BoxCountEnc_model.predict( BoxcountEncoder, opt.device,BatchToBePredicted, display)

        
        #MODE 2: USE LABELS FROM CPU BOXCOUNT
        if opt.Mode == 2:
            imgs, BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16 = BatchToBePredicted
            BCR_LAK_map_2.float(), BCR_LAK_map_4.float(), BCR_LAK_map_8.float(), BCR_LAK_map_16.float()

        real_labels_2 = Variable(BCR_LAK_map_2.type(Tensor))
        real_labels_2.to(opt.device)

        real_labels_4 = Variable(BCR_LAK_map_4.type(Tensor))
        real_labels_4.to(opt.device)

        real_labels_8 = Variable(BCR_LAK_map_8.type(Tensor))
        real_labels_8.to(opt.device)

        real_labels_16 = Variable(BCR_LAK_map_16.type(Tensor))
        real_labels_16.to(opt.device)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        valid.to(opt.device)
        fake.to(opt.device)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        real_imgs.to(opt.device)
        ori_imgs = Variable(imgs.type(Tensor))
        ori_imgs.to(opt.device)
        input_imgs = real_imgs[:,0,:,:].detach().cpu().numpy() #batch, 1, y,x

        # -----------------
        #  Test Generator
        # -----------------
        real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16 = inpainting(real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16)
        inpainted_imgs = real_imgs.detach().cpu().numpy() # for displaying purposes
        
        encoded_imgs = encoder(real_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16)

        decoded_imgs = decoder(encoded_imgs, real_labels_2, real_labels_4, real_labels_8, real_labels_16 )
        
        ###########################################################
        ###  SHOW GENERATED PICTURES
        ###########################################################

        ArrayList = np.array([])
        Namelist = []
        inpainted_imgs = inpainted_imgs[:,0,:,:]        
        gen_imgs = decoded_imgs[:,0,:,:].detach().cpu().numpy()
        for i in range(opt.batch_size):
            if i == 0:
                Namelist.append("Input")
                ArrayList = np.array([input_imgs[i,:,:]])
            else:
                ArrayList = np.append(ArrayList,[input_imgs[i,:,:]],axis=0)
                Namelist.append("")

        for i in range(opt.batch_size):
            ArrayList = np.append(ArrayList,[inpainted_imgs[i,:,:]],axis=0)
            if i == 0:
                Namelist.append("Noise/Mask")
            else:
                Namelist.append("")


        for i in range(opt.batch_size):
            try:
                ArrayList = np.append(ArrayList,[gen_imgs[i,:,:]],axis=0)
            except:
                pass
            if i == 0:
                Namelist.append("Output")
            else:
                Namelist.append("")

        showNPArrayImageArray(ArrayList, Namelist, opt, False)
        '''
        #clipping the gradients, cause AE_loss = nan happens # source: https://stackoverflow.com/questions/66648432/pytorch-test-loss-becoming-nan-after-some-iteration
        clip_value = 5
        torch.nn.utils.clip_grad_norm_(itertools.chain(encoder.parameters(), decoder.parameters()), clip_value)
        '''
        if opt.autoencoder == "off":
            # Loss measures generator's ability to fool the discriminator
            AE_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(decoded_imgs, ori_imgs)
            AE_loss[torch.isnan(AE_loss)] = 1.0
            #AE_loss[AE_loss == float("Inf")  ] = 1.0

            # ---------------------
            #  TEST Discriminator
            # ---------------------
            # Sample noise as discriminator ground truth
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim**2))))
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)

            d_loss = 0.5 * (real_loss + fake_loss)

            print("[[Batch %d/%d] [D loss: %f] [AE loss: %f] [G loss: %f]"
                % ( i, len(testDataloader), d_loss.item(), AE_loss.item(), fake_loss.item()))

            batches_done =  len(testDataloader) + i

        else:
            AE_loss = pixelwise_loss(decoded_imgs, ori_imgs)   
            AE_loss[torch.isnan(AE_loss)] = 1.0
            #AE_loss[AE_loss == float("Inf")  ] = 1.0         

            print(
                "[[Batch %d/%d] [AE loss: %f]"
                % ( i, len(testDataloader),  AE_loss.item())
            )

        control = input("What do you want to do? Next Batch(N), Break(b): (n/B)")
        if control.lower() == "n":
            continue
        elif control.lower() == "b" or control.lower() == "" :
            break


