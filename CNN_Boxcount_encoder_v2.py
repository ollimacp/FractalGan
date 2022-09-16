



#BOXCOUNT INIT
Boxsize=[2,4,8,16,32,64,128]    #,256,512,1024]
iteration = 0
#os.makedirs("images", exist_ok=True)

import os
#os.makedirs("images", exist_ok=True)
import numpy as np
import argparse

#what_do_you_want = input("Which function should be performed? 1: Rebuild data, 2: Train model, 3: Validate  ")


'''
# Create a option object to store all variables
class OptionObject:
  def __init__(self, n_epochs, batch_size, img_size, channels, learning_rate, b1, b2 ):
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.img_size = img_size
    self.lr = learning_rate
    self.b1 = b1   #first order momentum of gradient decay
    self.b2 = b2   #second order momentum of gradient decay
    self.channels = channels
    self.n_cpu = 8
    
#self, n_epochs, batch_size, img_size, channels, learning_rate, b1, b2
opt = OptionObject(100, 32, opt.img_size, 1 , 0.00002, 0.5, 0.8)
img_shape = (opt.channels, opt.img_size, opt.img_size)
'''



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--verbosity", type=bool, default=False, help="Set verbosity True/False for display results, or show additional data for debugging")

opt = parser.parse_args()
print(opt)



maxIndexY = opt.img_size
maxIndexX = opt.img_size

'''
import platform
Operating_system = platform.system()
print(Operating_system)
if Operating_system == 'Windows':
    print("Windows detected, dataloader multiprocessing not avaiable. Set n_cpu to 0 ")
    opt.n_cpu = 0
'''
#print("Manual set to opt.n_cpu=4")
#opt.n_cpu = 4

img_shape = (opt.channels, opt.img_size, opt.img_size)

shape = (opt.img_size, opt.img_size )


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
from PIL import Image
import gc
import math
import time
import BoxcountFeatureExtr_v2 as BoxcountFeatureExtr
import sklearn.preprocessing as preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials  #hyperoptimization libary

#import itertools
from PIL import Image
from os import listdir

# Common Directories
import pathlib              #Import pathlib to create a link to the directory where the file is at.
FileParentPath = str(pathlib.Path(__file__).parent.absolute())
saveplace = FileParentPath + "/Datasets/"


#Helper-function to show any np.array as a picture with a chosen title and a colormapping 
def showNPArrayAsImage(np2ddArray, title, colormap):
    plt.figure()                    #Init figure
    plt.imshow(np2ddArray,          #Gererate a picture from np.array and add to figure
            interpolation='none',
            cmap = colormap)
    plt.title(title)                #Add title to figure
    plt.show(block=False)           #Show array as picture on screen, but dont block the programm to continue.






#Source: [20] https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
#@jit(nopython=False)  #,forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def pad(array, reference, offset):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


#If a picture/array is more little than the reference shape, than add zeros to the right and bottom with pad()-function to bring it into opt.img_size x opt.img_size
def reshape_Data(PicNumPy,shape, original_shape):
    #if shape is bigger then original, set new max and retry making data
    reshape = (int(original_shape[0]),int(original_shape[1]) )
    if opt.verbosity:
        print("reshaping, cause shape and original shape are", shape, original_shape)
        print("reshape",reshape )
        print("PicNumPy shape",PicNumPy.shape)
    ### can add offset; so that pict can be centered
    offset = [0,0,0]
    PicNumPy = pad(PicNumPy,np.zeros(reshape),offset )
    return PicNumPy


#just functions via jupyter notebook with !command
def delete_dataset_from_last_time(FileParentPath):
    import shutil

    really = input("-->(y/n):  Do you want to delete the old Dataset? \n BE CARFUL: function can remove whole directorys, so dont change the Fileparentpath")
    
    
    if really =="y":
        OldDatasetSaveplace = FileParentPath+"/Datasets/test/"
        try:
            shutil.rmtree(OldDatasetSaveplace)
            os.mkdir(OldDatasetSaveplace)
            os.mkdir(OldDatasetSaveplace+"/features/")
            os.mkdir(OldDatasetSaveplace+"/labels/")
        except OSError:
            print("Deleting old test dataset failed")
            PrintException()
        else:
            print("Old test dataset deleted!")
        
        #!rm -rf OldDatasetSaveplace
        #!mkdir OldDatasetSaveplace
        #!mkdir OldDatasetSaveplace +"/features/"
        #!mkdir OldDatasetSaveplace +"/labels/"


        OldDatasetSaveplace = FileParentPath+"/Datasets/train/"
        try:
            shutil.rmtree(OldDatasetSaveplace)
            os.mkdir(OldDatasetSaveplace)
            os.mkdir(OldDatasetSaveplace+"/features/")
            os.mkdir(OldDatasetSaveplace+"/labels/")
        except OSError:
            print("Deleting old train dataset failed")        #!rm -rf OldDatasetSaveplace
            PrintException()
            input("Press any key to continue")
        else:
            print("Old train dataset deleted!")
        
        #!mkdir OldDatasetSaveplace
        #!mkdir OldDatasetSaveplace +"/features/"
        #!mkdir OldDatasetSaveplace +"/labels/"
    else:
        print("Continue without deleting the old dataset")
        input("Press any key to continue")


#Setting device to GPU/CPU 
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    return device

device = get_device()
print("Chosen Devide is",device)





class BoxCountEncoder(nn.Module):
    def __init__(self,Parameter):
        super(BoxCountEncoder, self).__init__()
        #Boxcount EncoderLayer<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.LayerDiscription = Parameter['BoxCountLayerDiscription']   #
        self.input_shape = Parameter['input_shape']
        self.LayerCount = len(self.LayerDiscription)
        self.Layers = nn.ModuleList() 
        self.OutputLayerIndexList = Parameter['OutputLayerIndexList']
        if opt.verbosity == True:
            print("self.OutputLayerIndexList", self.OutputLayerIndexList)

        #Throughput=[]   #list of layercount for Batchnormlayer
        #ATTENTION  ENHANCER BUILDS NETWORK BACKWARDS AND IN AND OUT ARE ALSO SWITCHED
        for i in range(self.LayerCount):        #iterate forwards
            IN,OUT,Kx, Ky,Sx,Sy,Px,Py,BN = self.LayerDiscription[i]
            if opt.verbosity: print("Layer",i," with Parameters", IN,OUT,Kx, Ky,Sx,Sy,Px,Py,BN)
            self.Layers.append(nn.Conv2d(IN,OUT, kernel_size=(Kx, Ky), stride=(Sx, Sy), padding=(Px, Py)) )      #Attention Compressor INOUT NOrmal
            
            #if this is an output layer, then use tanH instead of relu to prevent jumps for values around zero
            # tanh also allows scaled data, which it's mean is around zero and standard devitation of one
            if OUT==2 and  Kx==1 and Ky==1 and Sx==1 and Sy ==1 or  int(i) == int(self.LayerCount)-1: 
                if opt.verbosity: print("Output Layer found")
                #self.Layers.append(nn.Sigmoid())
                self.Layers.append(nn.Tanh())
                #Cause tanh is another layer it has to be in  self.OutputLayerIndexList, so it dosent connect to x but branched out to y
            else:
                #Just use LeakyReLU for fast convergence
                self.Layers.append(nn.LeakyReLU(inplace = True))

            if BN ==1:
                self.Layers.append( nn.BatchNorm2d(OUT, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) )      #Attention Compressor INOUT NOrmal
                #Throughput.append(OUT)      

        if opt.verbosity: print("self.input_shape",self.input_shape)
        print(self.Layers)
        print("-------------INIT DONE ------------------")



    def forward(self, x):

        output2, output4, output8, output16 = None, None, None, None
        OutputList = [output2, output4, output8, output16]
        outputindex = 0
        for i, layer in enumerate(self.Layers):        #iterate forwards
            #IN,OUT, Kx, Ky,Sx,Sy,Px,Py,BN = self.LayerDiscription[i]

            if i in self.OutputLayerIndexList:      #If this is a Output layer
                out = self.Layers[i](x)     # create branch and dont overwrite x
                #prints commented out, cause foreward will be executed millions of times
                #print("Create Branch")
            elif i-1 in self.OutputLayerIndexList:      #If this is a Output layer ACTIVATION FUNCTION (tanh)
                out = self.Layers[i](out)     # ACtivate out with act. fct
                OutputList[outputindex] = out       # set value for each output-layer / scale 
                #print("Activate Branch for output",outputindex)                
                outputindex +=1 
            else:
                x = self.Layers[i](x)
                #print("Append layer to Main Branch")

        return OutputList[0] , OutputList[1] , OutputList[2] , OutputList[3]







print("Imports and helper functions defined")



#   Data Balancing/Reshaping Part-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
The data has to be balanced in multiple ways to prevent overfitting.
For example not all pictures can be used to train data. 
Binary classes can be balanced just by taking 50/50 balance for the training dataset.
Cause the lables are the calculated arrays of a cpu driven program, 
there is just a continuum of output arrays for input arrays.
'''
opt.verbosity = False
precision = 1   # 0 dont balance, 1 balance light, ...9 balance fine; The finer, the more data will be discarded


ModelnameList = None

class CNN_BC_enc():

    def __init__(self,opt ):
        super(CNN_BC_enc, self).__init__()
        self.ModelnameList = None
    
    # opt.verbosity = False
    #INIT BoxcountNetParams as empty Dict
    #The BoxCountEncoder takes the image and calculates the boxcountratios and the Lakcountmaps for each iteration and passes all the arrays(iteration) and returns it.


    global previous_Best_Loss


    previous_Best_Loss = None
    Loss_Now = None


    def TrainSpacialBoxcount_with(self,HyperparameterAndENCODERCLASS):
        HyperParameterspace, BoxCountEncoder, previous_Best_Loss =  HyperparameterAndENCODERCLASS 

        # global previous_Best_Loss, Loss_Now
        #BoxCountEncoder = None

        #INIT VARIABLES
        opt.n_epochs = HyperParameterspace['n_epochs']
        opt.batch_size = HyperParameterspace['batch_size']
        opt.lr = HyperParameterspace['lr']
        opt.b1 = HyperParameterspace['b1']
        opt.b2 = HyperParameterspace['b2']

        #BoxCountLayerDiscription
        #self.BoxcountRatioConv = nn.Conv2d(3, 16, (self.BoxsizeX,BoxsizeY), (len(self.BoxsizeX), len(self.BoxsizeY) ), padding=0) 
        #self.LACcountConv = nn.Conv2d(3, 16, (self.BoxsizeX,self.BoxsizeY), (len(self.BoxsizeX), len(self.BoxsizeY) ), padding=0) 

        Boxsize=[2,4,8,16,32,64,128,256,512,1024]

        #Channles
        IN, OUTCOM = 1 , 2      #Channels 2, cause output is BCRmap and LAKmap, one is derived from the other
        Inter1, Inter2, Inter3    = HyperParameterspace['Inter1'], HyperParameterspace['Inter2'], HyperParameterspace['Inter3']          # Hyperoptimization here

        #Kernelsize in X/Y resprective layer is 2, cause...
        Kx1, Kx2, Kx3, Kx4              = 2,2,2,2       
        Ky1, Ky2, Ky3, Ky4              = 2,2,2,2
        # ...with a stride of 2 the picture is getting halfed in size, exacly like the boxcounting with bigger boxsizes , but in cpu version the ori bz overlap one pixel, which here is not checkk
        Sx1, Sx2, Sx3, Sx4              = 2, 2, 2, 2
        Sy1, Sy2, Sy3, Sy4              = 2, 2, 2, 2
        #padding should be 0, cause every picture is same size and all the kernels fit perfectly
        Px1, Px2, Px3, Px4              = 0,0,0,0
        Py1, Py2, Py3, Py4              = 0,0,0,0
        #Batchnorm is not needed, causewe want to focus just on the convolution calculation  and dont want to alter the image/ entry arrays in any unknown form... EVALUATE and CHECK 
        BN1, BN2, BN3, BN4              = 0,0,0,0

        #Intermediate OutputLayer ... Cause every Filter of a Convulution  is described within the Channels in the hidden layers and we want to output the BCR and LAK like like the cpu version...
        # we have to generate an output with a 1x1 = KxS conv with x Chan input and 2 Chan Output for calcing the loss

        BoxCountLayerDiscription = [    [IN,    Inter1,Kx1, Ky1,Sx1,Sy1,Px1,Py1,BN1],   #input layer
                                        [Inter1,OUTCOM ,1, 1,1,1,0,0,0],                # output layer for first iteration (Boxsize 2)

                                        [Inter1,Inter2,Kx2, Ky2,Sx2,Sy2,Px2,Py2,BN2],
                                        [Inter2,OUTCOM ,1, 1,1,1,0,0,0],                # output layer for second iteration (Boxsize 4)
            
                                        [Inter2,Inter3,Kx3, Ky3,Sx3,Sy3,Px3,Py3,BN3],
                                        [Inter3,OUTCOM ,1, 1,1,1,0,0,0],                # output layer for third iteration (Boxsize 8)

                                        [Inter3,OUTCOM,Kx4, Ky4,Sx4,Sy4,Px4,Py4,BN4],   #last ouput layer
                                                ]                       # [Inter3,OUTCOM ,1, 1,1,1,0,0,0],                # output layer for 4th iteration (Boxsize 16)

        input_shape = (opt.batch_size,1, opt.img_size,opt.img_size)

        OutputLayerIndexList = [2,6,10,12]        # Cause the intermediate output layers are branched out from the main flowchart

        BoxCountNetParameters = {'BoxCountLayerDiscription': BoxCountLayerDiscription, 'input_shape': input_shape, 'OutputLayerIndexList': OutputLayerIndexList}


        Modelname =  "n_epochs_" + str(round(opt.n_epochs,3)) 
        Modelname += "_batch-size_" + str(round(opt.batch_size,3))   
        Modelname += "_learning-rate_" + str(round(opt.lr,3))
        Modelname += "_beta-decay_" + str(round(opt.b1,3)) +"_" + str(round(opt.b2,3))
        #Modelname += "_Scalefactors_" + str(round(Scalefactor_2,3)) +"_" + str(round(Scalefactor_4,3)) +"_" + str(round(Scalefactor_8,3))


        # -----------------
        #  Train_BoxcountingCONV
        # -----------------

        #define Loss
        pixelwise_loss = torch.nn.L1Loss()

        #Init BoxcountEncoder
        BoxCountEncoder = BoxCountEncoder(BoxCountNetParameters)
        
        #print("try to use both gpus")
        #BoxCountEncoder = nn.DataParallel(BoxCountEncoder)


        BoxCountEncoder.to(device)
        pixelwise_loss.to(device)

        # Optimizers
        optimizer_BC = torch.optim.Adam(BoxCountEncoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        if device=="cuda":
            Tensor = torch.cuda.FloatTensor 

        else:
            Tensor = torch.FloatTensor

        #from tqdm import tqdm

        # ----------
        #  Training
        # ----------

        epochs =range(opt.n_epochs)

        print("Model: ",Modelname)
        TrailingLoss = 10.0
        LossLastRound = None
        for epoch in epochs:

            if opt.verbosity: print("epoch ",str(epoch), "of", str(opt.n_epochs) ) 
                
            for i, (images, labels_2, labels_4, labels_8, labels_16 ) in enumerate(self.trainDataloader):
                #real_labels_2, real_labels_4, real_labels_8, real_labels_16 = labels
                
                if opt.verbosity: print("Nr",str(i) ,"of", str(len(self.trainDataloader))) 
                real_labels_2 = Variable(labels_2.type(Tensor))
                real_labels_2.to(device)

                real_labels_4 = Variable(labels_4.type(Tensor))
                real_labels_4.to(device)

                real_labels_8 = Variable(labels_8.type(Tensor))
                real_labels_8.to(device)

                real_labels_16 = Variable(labels_16.type(Tensor))
                real_labels_16.to(device)

                # Configure input
                real_imgs = Variable(images.type(Tensor))
                real_imgs.to(device)

                optimizer_BC.zero_grad()

                BCR_LAK_map_2 , BCR_LAK_map_4 , BCR_LAK_map_8 , BCR_LAK_map_16 = BoxCountEncoder(real_imgs)


                BCR_LAK_map_2_loss  = pixelwise_loss(BCR_LAK_map_2, real_labels_2)
                BCR_LAK_map_4_loss = pixelwise_loss(BCR_LAK_map_4, real_labels_4)
                BCR_LAK_map_8_loss = pixelwise_loss(BCR_LAK_map_8, real_labels_8)
                BCR_LAK_map_16_loss = pixelwise_loss(BCR_LAK_map_16, real_labels_16)
                
                Scalefactor_2 , Scalefactor_4 , Scalefactor_8 , Scalefactor_16  =  0.25, 0.25, 0.25 , 0.25      # maybe optimiziing usage?!
                BCR_LAK_loss =  Scalefactor_2 * BCR_LAK_map_2_loss + Scalefactor_4 * BCR_LAK_map_4_loss + Scalefactor_8 * BCR_LAK_map_8_loss + Scalefactor_16 * BCR_LAK_map_16_loss
                if opt.verbosity: print("loss: BCR_LAK_loss",BCR_LAK_loss) 

                #input()
                BCR_LAK_loss.backward()
                optimizer_BC.step()
            
            print(
                "[Epoch %d/%d] [Batch %d/%d] [BC loss: %f]  "
                % (epoch, opt.n_epochs, i, len(self.trainDataloader), BCR_LAK_loss.item() )
            )

            #Cause some models dont converge we create a traling loss, which is the sum of the (trailing loss from the last round + the loss this round)/2 
            LossThisRound = float(BCR_LAK_loss.item())
            if LossThisRound == LossLastRound:
                #DoYouWantToBreak = input("Break? (Y/n): Check if model is converging, if Loss keeps to be the same, model isn't converging")
                #if DoYouWantToBreak == "" or DoYouWantToBreak.lower() == "y":
                break



            print("Last trailing Loss:", TrailingLoss)
            sumed = float(TrailingLoss+LossThisRound)
            TrailingLoss = np.divide(sumed,2.0)  

            LossLastRound = LossThisRound
            if LossThisRound >= TrailingLoss:
                #If The loss this round is Higher than the mean of the trailing loss, then break training, cause model isn't going anywhere 
                print("Breaking, cause model doesnt converge anymore, but please check anyway")
                break
            

            

        ### SAVE MODEL IF its better than 0something
        if previous_Best_Loss == None:
            previous_Best_Loss = BCR_LAK_loss.item()
        else:
            pass


        Loss_Now = BCR_LAK_loss.item()
        print("Best loss so far  :", previous_Best_Loss)
        print("loss of this model:", Loss_Now)

        if Loss_Now <= previous_Best_Loss:
            #<= to save first model always and then just, when better model was found with LOWER LOSS 
            saveplace = FileParentPath
            saveplace +="/models/"
            saveplace +="/SpacialBoxcountModels/"
            
            saveplace += "Loss" + str(round(BCR_LAK_loss.item(),3)) +"---"
            saveplace += Modelname
            NetParametersSaveplace = saveplace +".netparams"
            with open(NetParametersSaveplace, "wb") as f:
                pickle.dump(BoxCountNetParameters, f)
            
            saveplace += ".model"
            torch.save(BoxCountEncoder.state_dict(), saveplace)
            #only update, when it was higher
            print("Model Saved")
            previous_Best_Loss = Loss_Now

        else:
            print("Loss was higher/worse than previous best model")

        return {'loss': BCR_LAK_loss.item(), 'status': STATUS_OK}



    def begin_training(self, trainDataset, trainDataloader):
        #global trainDataset, trainDataloader
        self.trainDataset = trainDataset
        self.trainDataloader = trainDataloader

        #Source: https://github.com/hyperopt/hyperopt/issues/267
        #To save trials object to pick up where you left
        def run_trials(HyperParameterspace, Modelname):
            #ATTENTION: If you want to begin training anew, then you have to delete the .hyperopt file
            TrialsSaveplace = FileParentPath
            TrialsSaveplace +=  "/"+ str(Modelname) +".hyperopt" 
            trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
            max_trials = 3  # initial max_trials. put something small to not have to wait

            
            try:  # try to load an already saved trials object, and increase the max
                trials = pickle.load(open(TrialsSaveplace, "rb"))
                print("Found saved Trials! Loading...")
                max_trials = len(trials.trials) + trials_step
                print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
            except:  # create a new trials object and start searching
                trials = Trials()

            
            lowest_loss = fmin(self.TrainSpacialBoxcount_with, HyperparameterAndENCODERCLASS, algo=tpe.suggest, max_evals=max_trials, trials=trials)

            print("Lowest achieved loss so far:", lowest_loss)
            
            # save the trials object
            with open(TrialsSaveplace, "wb") as f:
                pickle.dump(trials, f)

        #old batchsize list  [2,4,8,16,32,64,128,256,512]

        HyperParameterspace = {
            'n_epochs':hp.choice('opt.n_epochs', range(5,150,5) ),
            'batch_size':hp.choice('opt.batch_size', [2,4,8,16,32,64,128,256,512,1024,2048,4096] ),
            'lr':hp.uniform('lr', 0.0000001 , 0.1 ),
            'b1':hp.uniform('b1', 0.01 , 1.0 ),
            'b2':hp.uniform('b2', 0.01 , 1.0 ),
            'Inter1':hp.choice('Inter1', range(1,512) ),
            'Inter2':hp.choice('Inter2', range(1,512) ),
            'Inter3':hp.choice('Inter3', range(1,512) ),
            #'Scalefactor_2':hp.uniform('Scalefactor_2', 0.4, 0.5 ),   
            #'Scalefactor_4':hp.uniform('Scalefactor_4', 0.15, 0.25 ),
            #'Scalefactor_8':hp.uniform('Scalefactor_8', 0.1, 0.15 ),

        }     

        HyperparameterAndENCODERCLASS = HyperParameterspace, BoxCountEncoder,previous_Best_Loss

        print("Begin HyperparameterOptimization")

        # loop indefinitely and stop whenever you like by setting MaxTrys

        #TotalTrials = 0 
        MaxTrys = 100
        for TotalTrials in range(MaxTrys):
            Modelname = "SpacialBoxcountEncoder"+"_trialsOBJ"
            run_trials(HyperparameterAndENCODERCLASS, "BoxcountEncoder")    

    

    def validation(self,testDataset, testDataLoader):
        if self.ModelnameList == None:
            # -------------------------------------------------------------------------------------------------------
            #  Testing BoxcountEncoder
            # -------------------------------------------------------------------------------------------------------
            self.ModelnameList = []
            #Pretrained Networks-----------------------
            self.ModelnameList.append("Loss0.529---n_epochs_90_batch-size_512_learning-rate_0.063_beta-decay_0.928_0.664")
            self.ModelnameList.append("Loss0.01---n_epochs_135_batch-size_4_learning-rate_0.001_beta-decay_0.671_0.362")
            self.ModelnameList.append("Loss0.01---n_epochs_85_batch-size_512_learning-rate_0.001_beta-decay_0.681_0.876")
            self.ModelnameList.append("Loss0.014---n_epochs_5_batch-size_128_learning-rate_0.001_beta-decay_0.501_0.945")
            self.ModelnameList.append("Loss0.506---n_epochs_25_batch-size_512_learning-rate_0.071_beta-decay_0.26_0.248")
            self.ModelnameList.append("Loss0.735---n_epochs_95_batch-size_4_learning-rate_0.062_beta-decay_0.311_0.194")
            self.ModelnameList.append("Loss0.012---n_epochs_135_batch-size_4_learning-rate_0.015_beta-decay_0.808_0.762")

        showitem = None
        whereTObreakIteration = 100   #at batch the test will break the testloop to continue


        #To render the network output set verbosity to True
        opt.verbosity = True
        opt.n_cpu = 8 #for every thread of my quadcore -> adjust as you like

        def TestSpacialBoxcount_with(Modelname, BoxCountEncoder,showitem, device):
            #global showitem
            NetParametersSaveplace =FileParentPath+ "/models/"+ "SpacialBoxcountModels/"+ Modelname +".netparams"
            BoxCountNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))

            saveplace = FileParentPath+ "/models/"+ "SpacialBoxcountModels/"+Modelname +".model"
            
            __, parameter = Modelname.split('---')  #extracting parameter to generate option object
            __, __, n_epochs, __,   batch_size, __, learning_rate, __ , betadecay1, betadecay2 = parameter.split('_')
            
            
            #should not be nessecary, except the changing batchsize #self, n_epochs, batch_size, img_size, channels, learning_rate, b1, b2
            #opt = OptionObject(int(n_epochs), int(batch_size), opt.img_size, 1 , float(learning_rate), float(betadecay1), float(betadecay2))
            
            #device = "cuda"
            #device = "cpu"
            
            #load and init the BCencoder
            BoxCountEncoder = BoxCountEncoder(BoxCountNetParameters)
            try:
                BoxCountEncoder.load_state_dict(torch.load(saveplace, map_location=device))
            except:
                BoxCountEncoder.load_state_dict(torch.jit.load(saveplace, map_location=device))

            BoxCountEncoder.eval()   #to disable backpropagation, so don't adjust any weights and biases

            #define Loss
            pixelwise_loss = torch.nn.L1Loss()

            #Init BoxcountEncoder
            BoxCountEncoder.to(device)
            pixelwise_loss.to(device)

            # Optimizers
            optimizer_BC = torch.optim.Adam(BoxCountEncoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

            if device=="cuda":
                Tensor = torch.cuda.FloatTensor 
            else:
                Tensor = torch.FloatTensor
            
            totaltime = 0.0
            running_loss = 0.0
            
            
            #Begin testing by evaluating the test data set
            for  i, (images, labels_2, labels_4, labels_8, labels_16 )  in enumerate(testDataLoader):
                if i== whereTObreakIteration:
                    print("reached iteration",whereTObreakIteration, "break loop to test next model" )
                    break
                    
                start = time.time()
                
                torch.no_grad() # for testing no gradients have to be computed
                # Configure input/ouput variables and send it to device
                real_imgs = Variable(images.type(Tensor))
                real_imgs.to(device)
                # BCR_map/Lac_map with boxsize 2
                real_labels_2 = Variable(labels_2.type(Tensor))
                real_labels_2.to(device)
                # BCR_map/Lac_map with boxsize 4
                real_labels_4 = Variable(labels_4.type(Tensor))
                real_labels_4.to(device)
                # BCR_map/Lac_map with boxsize 8
                real_labels_8 = Variable(labels_8.type(Tensor))
                real_labels_8.to(device)
                # BCR_map/Lac_map with boxsize 16
                real_labels_16 = Variable(labels_16.type(Tensor))
                real_labels_16.to(device)

                BCR_LAK_map_2 , BCR_LAK_map_4 , BCR_LAK_map_8 , BCR_LAK_map_16 = BoxCountEncoder(real_imgs)
                optimizer_BC.zero_grad()
                
                #necessary for accessing the picture again with numpy
                NumpyencImg2 =  BCR_LAK_map_2.cpu().detach().numpy()
                NumpyencImg4 =  BCR_LAK_map_4.cpu().detach().numpy()
                NumpyencImg8 =  BCR_LAK_map_8.cpu().detach().numpy()
                NumpyencImg16 =  BCR_LAK_map_16.cpu().detach().numpy()

                if opt.verbosity:
                    #Render Pictures 
                    
                    for idx in range(opt.batch_size):
                        print("index within batch:", idx)
                        #CreateSubplotWith(idx, images, labels_2, labels_4, labels_8, labels_16,NumpyencImg2, NumpyencImg4, NumpyencImg8, NumpyencImg16)

                        showNPArrayAsImage(images[idx,0,:,:], "Original Image", "gray")

                        #source: https://stackoverflow.com/questions/22053274/grid-of-images-in-matplotlib-with-no-padding
                        max_cols = 8
                        fig, axes = plt.subplots(nrows=2, ncols=max_cols, figsize=(16,4))
                        lablelist = [labels_2[idx,0,:,:], labels_2[idx,1,:,:], labels_4[idx,0,:,:], labels_4[idx,1,:,:], labels_8[idx,0,:,:], labels_8[idx,1,:,:], labels_16[idx,0,:,:],  labels_16[idx,1,:,:], NumpyencImg2[idx,0,:,:], NumpyencImg2[idx,1,:,:], NumpyencImg4[idx,0,:,:], NumpyencImg4[idx,1,:,:], NumpyencImg8[idx,0,:,:], NumpyencImg8[idx,1,:,:], NumpyencImg16[idx,0,:,:]  , NumpyencImg16[idx,1,:,:]]
                        #titlelist = ["BCR2", "LAC2", "BCR4", "LAC4", "BCR8", "LAC8","BCR16", "LAC16", "NN BCR2", "NN LAC2",   ]
                        ylabellist = ["CPU", "", "","", "", "", "","", "GPU", "","","","", "", "",""]
                        xlabellist = ["", "", "","", "", "","","","BCR2" ,"LAC2" ,"BCR4" ,"LAC4" ,"BCR8" ,"LAC8" ,"BCR16" ,"LAC16" ]
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
                        plt.show(block=True)

                        if showitem == None or showitem == "y" or showitem == "Y":
                            #showitem = input("press  y/Y for next item in batch  or else to continue with next batch")
                            print("input not possible with multicore just show next batchso showitem is set manually to no")
                            showitem = "n"
                            if showitem == "y" or showitem == "Y":
                                continue
                            else:
                                showitem = None
                                break

                            
                end = time.time()  #cause loss calulation has nothing to do with boxcount calc time

                # Scalable pixelwise loss to have    
                BCR_LAK_map_2_loss  = pixelwise_loss(BCR_LAK_map_2, real_labels_2)
                BCR_LAK_map_4_loss = pixelwise_loss(BCR_LAK_map_4, real_labels_4)
                BCR_LAK_map_8_loss = pixelwise_loss(BCR_LAK_map_8, real_labels_8)
                BCR_LAK_map_16_loss = pixelwise_loss(BCR_LAK_map_16, real_labels_16)
                Scalefactor_2 , Scalefactor_4 , Scalefactor_8 , Scalefactor_16  =  0.25, 0.25, 0.25 , 0.25      # maybe optimiziing usage?!
                #assert Scalefactor_2 + Scalefactor_4 + Scalefactor_8 + Scalefactor_16 == 1.0

                BCR_LAK_loss =  Scalefactor_2 * BCR_LAK_map_2_loss + Scalefactor_4 * BCR_LAK_map_4_loss + Scalefactor_8 * BCR_LAK_map_8_loss + Scalefactor_16 * BCR_LAK_map_16_loss
                
                running_loss += BCR_LAK_loss.item()

                BCR_LAK_loss.backward()
                optimizer_BC.step()
                
                timePERbatch = end - start
                totaltime += timePERbatch
                
                if opt.verbosity: 
                    print("[Batch %d/%d] [BC loss: %f] "% ( i, len(testDataLoader), BCR_LAK_loss.item()))
                    print(timePERbatch, " seconds for boxcounting 1 file with batch_size of",opt.batch_size )
                    
                #input("presskey fornext batch")
            mean_timePERbatch = totaltime / float(whereTObreakIteration)
            mean_loss = running_loss/ float(whereTObreakIteration)  # cause testing will abort after 100 batchesm has to be normalized 

            return mean_loss, mean_timePERbatch




        scoreboard = {}
        #{'Modelname': mean_loss, timePerbatch, MegapixelPERsecond}

        #If cuda is avaiable, then test against both
        device = get_device()
        if device=="cuda":
            devicelist = ['cpu','cuda']
        else:
            devicelist = ['cpu']

        for device in devicelist:
            
            for Modelname in self.ModelnameList:
                print("-----------begin test with new model-------------")
                print("Chosen Device is", device)
                print("Modelname: ", Modelname)
                mean_loss,  mean_timePERbatch = TestSpacialBoxcount_with(Modelname,BoxCountEncoder,showitem, device)
                mean_loss,  mean_timePERbatch = round(mean_loss,2),  round(mean_timePERbatch,2) 
                MegapixelPERsecond =  round( (opt.batch_size * opt.img_size**2) /(mean_timePERbatch* 1000000 ) ,2)
                fullmodelname = device +'_'+ Modelname
                scoreboard[fullmodelname] = [mean_loss, mean_timePERbatch, MegapixelPERsecond ]
                print("This model performed with a mean loss of", mean_loss, "with a mean time/batch", mean_timePERbatch, "with a pixelthroughput of",MegapixelPERsecond," Mpx/s" )

        #print("scoreboard: ", scoreboard)
        for key, item in scoreboard.items():
            print("Model:",key," with mean_testloss of", item[0], " with a mean time/batch", item[1], "with a pixelthroughput of", item[2] )



    def predict(self,BoxCountEncoder ,device,BatchToBePredicted, display):
        #To render the network output set verbosity to True
        opt.verbosity = True
        '''
        #BoxCountEncoder =
        #Modelname = "Loss0.014---n_epochs_5_batch-size_128_learning-rate_0.001_beta-decay_0.501_0.945"
        #print("BoxCountEncoder")
        #input("alidjh")
        #opt.n_cpu = 8 #for every thread of my quadcore -> adjust as you like

        NetParametersSaveplace =FileParentPath+ "/models/"+ Modelname +".netparams"
        BoxCountNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))

        saveplace = FileParentPath+ "/models/"+Modelname +".model"
        
        __, parameter = Modelname.split('---')  #extracting parameter to generate option object
        __, __, n_epochs, __,   batch_size, __, learning_rate, __ , betadecay1, betadecay2 = parameter.split('_')
        
        
        #should not be nessecary, except the changing batchsize #self, n_epochs, batch_size, img_size, channels, learning_rate, b1, b2
        #opt = OptionObject(int(n_epochs), int(batch_size), opt.img_size, 1 , float(learning_rate), float(betadecay1), float(betadecay2))
        
        #device = "cuda"
        #device = "cpu"
        
        #load and init the BCencoder
        #BoxCountEncoder = BoxCountEncoder(BoxCountNetParameters)
        try:
            BoxCountEncoder.load_state_dict(torch.load(saveplace, map_location=device))
        except:
            BoxCountEncoder.load_state_dict(torch.jit.load(saveplace, map_location=device))
        #BoxCountEncoder.load_state_dict(torch.load(saveplace, map_location=device))
        BoxCountEncoder.eval()   #to disable backpropagation, so don't adjust any weights and biases

        #define Loss
        pixelwise_loss = torch.nn.L1Loss()

        #Init BoxcountEncoder
        #MAYBE NOT NESSECARY CAUSE OF map_location=device ---> Test
        BoxCountEncoder.to(device)
        pixelwise_loss.to(device)

        # Optimizers
        optimizer_BC = torch.optim.Adam(BoxCountEncoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


        #totaltime = 0.0
        #running_loss = 0.0
        
        '''
        if device=="cuda":
            Tensor = torch.cuda.FloatTensor 
        else:
            Tensor = torch.FloatTensor
                
        #Begin testing by evaluating the test data set
        images, labels_2, labels_4, labels_8, labels_16 = BatchToBePredicted

        print("images size",images.size())
            
        #maybe not NESSECARY cause model.eval() --- Test !!
        torch.no_grad() # for testing no gradients have to be computed

        # Configure input/ouput variables and send it to device
        real_imgs = Variable(images.type(Tensor))
        real_imgs.to(device)
        '''
        # BCR_map/Lac_map with boxsize 2
        real_labels_2 = Variable(labels_2.type(Tensor))
        real_labels_2.to(device)
        # BCR_map/Lac_map with boxsize 4
        real_labels_4 = Variable(labels_4.type(Tensor))
        real_labels_4.to(device)
        # BCR_map/Lac_map with boxsize 8
        real_labels_8 = Variable(labels_8.type(Tensor))
        real_labels_8.to(device)
        # BCR_map/Lac_map with boxsize 16
        real_labels_16 = Variable(labels_16.type(Tensor))
        real_labels_16.to(device)
        '''
        BCR_LAK_map_2 , BCR_LAK_map_4 , BCR_LAK_map_8 , BCR_LAK_map_16 = BoxCountEncoder(real_imgs)
        #optimizer_BC.zero_grad()
        
        #necessary for accessing the picture again with numpy
        NumpyencImg2 =  BCR_LAK_map_2.cpu().detach().numpy()
        NumpyencImg4 =  BCR_LAK_map_4.cpu().detach().numpy()
        NumpyencImg8 =  BCR_LAK_map_8.cpu().detach().numpy()
        NumpyencImg16 =  BCR_LAK_map_16.cpu().detach().numpy()
        
        showitem = None

        #opt.verbosity = True
        
        if display:
            #Render Pictures 
            for idx in range(opt.batch_size):
                print("index within batch:", idx)
                #CreateSubplotWith(idx, images, labels_2, labels_4, labels_8, labels_16,NumpyencImg2, NumpyencImg4, NumpyencImg8, NumpyencImg16)

                showNPArrayAsImage(images[idx,0,:,:], "Original Image", "gray")

                #source: https://stackoverflow.com/questions/22053274/grid-of-images-in-matplotlib-with-no-padding
                max_cols = 8
                fig, axes = plt.subplots(nrows=2, ncols=max_cols, figsize=(16,4))
                lablelist = [labels_2[idx,0,:,:], labels_2[idx,1,:,:], labels_4[idx,0,:,:], labels_4[idx,1,:,:], labels_8[idx,0,:,:], labels_8[idx,1,:,:], labels_16[idx,0,:,:],  labels_16[idx,1,:,:], NumpyencImg2[idx,0,:,:], NumpyencImg2[idx,1,:,:], NumpyencImg4[idx,0,:,:], NumpyencImg4[idx,1,:,:], NumpyencImg8[idx,0,:,:], NumpyencImg8[idx,1,:,:], NumpyencImg16[idx,0,:,:]  , NumpyencImg16[idx,1,:,:]]
                #titlelist = ["BCR2", "LAC2", "BCR4", "LAC4", "BCR8", "LAC8","BCR16", "LAC16", "NN BCR2", "NN LAC2",   ]
                ylabellist = ["CPU", "", "","", "", "", "","", "GPU", "","","","", "", "",""]
                xlabellist = ["", "", "","", "", "","","","BCR2" ,"LAC2" ,"BCR4" ,"LAC4" ,"BCR8" ,"LAC8" ,"BCR16" ,"LAC16" ]
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
                plt.show(block=True)

                if showitem == None or showitem == "y" or showitem == "Y":
                    try:
                        showitem = input("press  y/Y for next item in batch  or else to continue with next batch")
                    except:
                        PrintException()
                        print("Just waiting 2 s and proceed to next batch")
                        showitem = ""
                        time.sleep(2)
                    if showitem == "y" or showitem == "Y":
                        continue
                    else:
                        showitem = None
                        break

        

        return NumpyencImg2, NumpyencImg4, NumpyencImg8, NumpyencImg16



    #TrainBoxcountEncoder = False




    '''
    https://stackoverflow.com/questions/58296345/convert-3d-tensor-to-4d-tensor-in-pytorch
    x = torch.zeros((4,4,4))   # Create 3D tensor 
    print(x[None].shape)       #  (1,4,4,4)
    print(x[:,None,:,:].shape) #  (4,1,4,4)
    print(x[:,:,None,:].shape) #  (4,4,1,4)
    print(x[:,:,:,None].shape) #  (4,4,4,1)

    train_data = train_data[:,None,:,:]
    test_data = test_data[:,None,:,:]

    print("train_data.shape", train_data.shape,"test_data.shape", test_data.shape)


    '''





    
'''
#if returns True, dataset stays balanced, so take into train-data, else pack into test set. 
#@jit(nopython=False)  #,forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def CalcPlacingcondition(self, DensityMap, sumBCR,sumLAK, precision, lastVariance, index):
    #print("DensityMap",DensityMap)
    print("DensityMap.shape",DensityMap.shape)

    element = np.array([[sumBCR,sumLAK],])
    
    if index ==0:
        combinedDensityMap = element     #init first element
    else:
        #concatenate the BCR and LAK from the current element to the 
        combinedDensityMap = np.concatenate((DensityMap,element),axis=0)

    
    if precision == 0:
        placingcondition = True
        DensityMap = combinedDensityMap
        lastVariance = np.var(combinedDensityMap)
    else:
        if index <= 6:
            #to populate the field, just add the first 6 elements
            placingcondition = True

            #cause element placingcondition is true, update the variances             
            combinedVariance = np.var(combinedDensityMap)
            lastVariance =  combinedVariance
            print("populate with minimum Pop")
        else:
            #calculate the Variance from this turn
            combinedVariance = np.var(combinedDensityMap)

            #if the rounded variance of this turn is more or the same of the variance last turn
            if round(combinedVariance,precision) >= round(lastVariance,precision):    
                placingcondition = True
                #and update the last variance for next turn with the new value
                lastVariance =  combinedVariance
                DensityMap = combinedDensityMap
            else:
                #dont add this element to the training dataset but to the test dataset
                placingcondition = False

            if opt.verbosity: print("index:", index,"    formerVariance,combinedVariance",round(lastVariance,precision),round(combinedVariance,precision),"so placing is",placingcondition)    
            #input()

    return placingcondition, DensityMap, lastVariance



def make_train_test_data(self, shape):
    
    Num_test = 0
    Num_train = 0
    firsttime = True

    DensityMap= np.array([[0.0,0.0],])
    lastVariance = 0.0
    original_shape = shape

    #For Visualizing  tqdm-progress bar
    from tqdm import tqdm
    pbar = tqdm(total=maxIndexY-1)
    counter = 0
    train_counter = 0
    test_counter =0

    from os import listdir
    DataFolder = FileParentPath + "/data/Images/"
    filelist = [f for f in listdir(DataFolder)]
    #print(filelist)



    from tqdm import tqdm
    pbar = tqdm(total=len(filelist))

    
    n = 4 # werden wohl bis max 16 begrenzen # ist 4 da beim slicen  richitg wäre  boxsize[:4] oder [:-7]   

    for index, filename in enumerate(filelist):
        pbar.update(1)  #updates progressbar
        counter +=1
        try:   
            filepath = DataFolder+ filename  #Load Image with Pillow
            # Open the image
            image = Image.open(filepath)
            
            # If the picture is in RGB or other multi-channel mode 
            #just seperate the channels and concatenate them from left to right
            ChannleDimension = len(str(image.mode)) # grey -> 1 chan , rgb 3 channle
            
            if opt.verbosity == True:
                # summarize some details about the image
                print(image.format,image.size)
                print(image.mode)
                #show the image
                image.show()
                print("ChannleDimension",ChannleDimension)

            channelcodierung = []
            for channel in image.mode:
                #FLATTEN EACH CHANNEL TO ONE  BY TILLING, cause cnn have to be consistent channles
                #and if one rgb is in grayscale, then error
                if opt.verbosity: print(channel)
                channelcodierung.append(channel)
            
            C1, C2, C3, C4, C5, C6 = None, None, None, None, None, None
            channellist = [C1,C2,C3,C4,C5,C6]
            croppedChannelList = channellist[0:ChannleDimension-1]
            croppedChannelList = image.split()        
            initEntry = None
            stackedchannels = np.array(initEntry)
            for idx, channel in enumerate(croppedChannelList):
                PicNumpy = np.array(channel)
                if opt.verbosity == True:
                    print(PicNumpy)
                    print(PicNumpy.shape)

                if idx == 0:
                    stackedchannels = PicNumpy
                else:
                    stackedchannels = np.concatenate((stackedchannels,PicNumpy),axis=1)
                
            
            if opt.verbosity: print(stackedchannels.shape)
            npOutputFile = stackedchannels

            
            #########################################################################
            # MULTITHREAD BOXCOUNT LABLE EXTRACTION
            BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadBoxcount(npOutputFile)

            
            ##### CALC PLACING CONDITION FOR EACH DATAPOINT(PICTURE)
            maxiteration = 3 # cause boxsize 2,4,8,16 = 0,1,2,3 iteration # determines the maximum boxsize
            #breaking it down to 2 values
            sumBCR, sumLAK = BoxCountR_SpacialLac_map_Dict[1][0], BoxCountR_SpacialLac_map_Dict[1][1]             #BoxCountR_SpacialLac_map_Dict[4] #[0], BoxCountR_SpacialLac_map_Dict[4][1] 
            if opt.verbosity: print("BCR,sumLAK  MAP before sum", sumBCR,sumLAK)

            sumBCR, sumLAK = np.sum(sumBCR), np.sum(sumLAK)
            if opt.verbosity: print("sumBCR,sumLAK after sum", sumBCR,sumLAK)
            
            #Calc, if the next data in the dataset will balance it more or not
            placingcondition, DensityMap, lastVariance = self.CalcPlacingcondition(DensityMap, sumBCR,sumLAK,precision,lastVariance, index)

            maxChunkCount = (npOutputFile.shape[1]/opt.img_size) * (npOutputFile.shape[0]/opt.img_size)   #Chunks the picture is broken down into
            #print("maxChunkCount", maxChunkCount)
            Chunks = [None] * int(np.ceil(maxChunkCount))     # round up the non full boxes, cause they will be reshaped by padding with zeros       
            start = time.time()

            BoxBoundriesY = [0,opt.img_size]
            BoxBoundriesX = [0,opt.img_size]

            iteration = 0       # is Boxsize 32 -> should be faster then more little boxsize
            Boxsize=[2,4,8,16,32,64,128,256,512,1024]
            scalingFaktor = 1.0 / float(Boxsize[iteration])
            #If we take a sclice from the BCRmap/LAKmap, the boxboundries have to  be scaled for maintaining spacial dimensions across scaling with iteration and Boxsize
            Scaled_BoxBoundriesY = [0,int(opt.img_size*scalingFaktor)]
            Scaled_BoxBoundriesX = [0,int(opt.img_size*scalingFaktor)]


            for i in range(len(Chunks)):

                Chunks[i] = npOutputFile[BoxBoundriesY[0]:BoxBoundriesY[1],BoxBoundriesX[0]:BoxBoundriesX[1]] 

                CHUNKED_BoxCountR_SpacialLac_map_Dict = {}
                CuttedBoxsizeList = Boxsize[:maxiteration+1]
                if opt.verbosity == True:
                    print("Current box",i,"of all",len(Chunks),"boxes")
                    print("Boxboundries: X, Y :" ,BoxBoundriesX,BoxBoundriesY)
                    print("CuttedBoxsizeList", CuttedBoxsizeList)
                    print("Converting BCRmap,LAKmap into Chunked Form")
                    
                for it , currentboxsize in enumerate(CuttedBoxsizeList):
                    if opt.verbosity: print("Iteration", it, " and currentBoxsize", currentboxsize)
                    
                    scalingFaktor = 1.0 / float(currentboxsize)
                    
                    try:
                        #calc Scaled BoxBoundriesY
                        Scaled_BoxBoundriesY = [int(BoxBoundriesY[0]*scalingFaktor),int(BoxBoundriesY[1]*scalingFaktor)]

                    except:
                        PrintException()
                        #Assuming devide by zero
                        Scaled_BoxBoundriesY = [0,int(BoxBoundriesY[1]*scalingFaktor)]

                    try:
                        #calc Scaled BoxBoundriesX
                        Scaled_BoxBoundriesX = [int(BoxBoundriesX[0]*scalingFaktor),int(BoxBoundriesX[1]*scalingFaktor)]

                    except:
                        PrintException()
                        #Assuming devide by zero
                        Scaled_BoxBoundriesX = [0,int(BoxBoundriesX[1]*scalingFaktor)]                       


                    BCRmap, LAKmap = BoxCountR_SpacialLac_map_Dict[it]
                    
                    chunked_BCRmap = BCRmap[Scaled_BoxBoundriesY[0]:Scaled_BoxBoundriesY[1],Scaled_BoxBoundriesX[0]:Scaled_BoxBoundriesX[1]] 
                    chunked_LAKmap = LAKmap[Scaled_BoxBoundriesY[0]:Scaled_BoxBoundriesY[1],Scaled_BoxBoundriesX[0]:Scaled_BoxBoundriesX[1]] 
                    
                    #SCALING----------------------------
                    #Scale the values of the Arrays in a gaussian distrubution with mean 0 and diviation 1?!?!?!
                    #chunked_BCRmap, chunked_LAKmap = preprocessing.scale(chunked_BCRmap) , preprocessing.scale(chunked_LAKmap) 

                    
                    #ATTENTION SCALING DEACTIVATED
                    
                    
                    #Normalize the Values betweeen -1...1----------------------------
                    chunked_BCRmap, chunked_LAKmap = preprocessing.normalize(chunked_BCRmap, norm='l1')  , preprocessing.normalize(chunked_LAKmap, norm='l1')  


                    if opt.verbosity == True:
                        showNPArrayAsImage(Chunks[i], "Current Chunk", "gray")
                        print("chunked_BCRmap",chunked_BCRmap)
                        print("chunked_LAKmap",chunked_LAKmap)
                        showNPArrayAsImage(chunked_BCRmap, "chunked_BCRmap", "gray")
                        showNPArrayAsImage(chunked_LAKmap, "LAKmap", "gray")                    
                    
                    #index the BCR /LAK map to the right size 
                    CHUNKED_BoxCountR_SpacialLac_map_Dict[it] = [chunked_BCRmap, chunked_LAKmap]


                #Scale Dataset : Scaled data has zero mean and unit variance
                #Chunks[i]= preprocessing.scale(Chunks[i])
                
                
                #ATTENTION SCALING DEACTIVATED
                
                
                
                #Normalizing ARRAY  from 0...255 to -1...+1
                Chunks[i] = preprocessing.normalize(Chunks[i], norm='l1')        
                Chunkshape = Chunks[i].shape

                if BoxBoundriesX[1] > npOutputFile.shape[1]  or BoxBoundriesY[1] > npOutputFile.shape[0]:
                    if opt.verbosity: print("Chunkshape and shape are Diffrent... reshaping")
                    Chunks[i] =  reshape_Data(Chunks[i], Chunkshape, shape)
                    continue

                assert Chunks[i].shape == original_shape

                newshape = ( 1,int(original_shape[0]),int(original_shape[1]) )
                Chunks[i] = np.reshape(Chunks[i], newshape)

                #saving test or Train image and label
                feature =  Chunks[i]
                label =  [np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[0]) , np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[1]) , np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[2]) , np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[3]) ]
                
                if opt.verbosity == True:
                    print("feature",feature)
                    print("label",label)

                if placingcondition == True:
                    if opt.verbosity: print("Placingcondition is true, Append Chunk to dataset")
                    #saveplace
                    trainsaveplace = FileParentPath+"/Datasets/train/"
                    #saving image
                    imagesaveplace = trainsaveplace+ "/features/"+"Feature"+ str(Num_train)
                    np.save(imagesaveplace, feature)
                    #saving label
                    labelsaveplace = trainsaveplace+ "/labels/"+"label"+ str(Num_train)
                    #CANT save List with diffrent sized np arrays with np.save -> use pickle as workaround
                    pickle.dump(label,open(labelsaveplace,"wb"))
                    Num_train +=1
                        
                else:
                    testsaveplace = FileParentPath+"/Datasets/test/"
                    #saving image
                    imagesaveplace = testsaveplace+ "/features/"+"Feature"+ str(Num_test)
                    np.save(imagesaveplace, feature)
                    #saving label
                    labelsaveplace = testsaveplace+ "/labels/"+"label"+ str(Num_test)
                    #np.save(labelsaveplace, label)
                    #CANT save List with diffrent sized np arrays with np.save -> use pickle as workaround
                    pickle.dump(label,open(labelsaveplace,"wb"))
                    Num_test +=1



                #After this Chunk set the new Borders of the new chunk for next turn

                if BoxBoundriesX[1] < npOutputFile.shape[1]:

                    BoxBoundriesX[0] =BoxBoundriesX[0] + maxIndexX
                    BoxBoundriesX[1] = BoxBoundriesX[1] + maxIndexX
                    if opt.verbosity == True: 
                        print("Move box into x direction")
                        print("BoxBoundriesX", BoxBoundriesX)
                else:
                    if opt.verbosity == True:
                        print(BoxBoundriesY,"BoxBoundriesY") 
                        print("move box into starting position in x-direction")
                        print("move box into ydirection")
                    BoxBoundriesX[0]=0
                    BoxBoundriesX[1]=maxIndexX
                    BoxBoundriesY[0]+=maxIndexY
                    BoxBoundriesY[1]+=maxIndexY
                    
            if opt.verbosity: input("Press any key for next File")


            end = time.time()     
            print(round(end,1) - round(start,1), "seconds passed for chunking and Make Train Data for 1 File")

        except :
            PrintException()
            input()
            pass
    
    pbar.close()        #close the percentage bar cause all pictures have been processed

    if opt.verbosity:
        #To evaluate, if balancing happened  show figure of all BCR/LAKS of the training dataset
        x,y = np.array([]) , np.array([])
        for Koordinate in DensityMap:
            x,y = np.append(x,Koordinate[0]) , np.append(y,Koordinate[1])

        # Plot
        plt.scatter(x, y,s=1, alpha=0.33)
        plt.title('Lacunarity-Boxcountratio Diagramm')
        plt.xlabel('sumBCR')
        plt.ylabel('MeanLAK')
        plt.show()

    print("Num_train",Num_train ,"Num_test",Num_test)




def create_dataset(self,shape):
    delete_dataset_from_last_time(FileParentPath)
    print("Begin preprocessing train/test datasets")
    self.make_train_test_data(shape)
    print("Datasets created")
    #REBUILDING/Balancing DATA DONE ---------------------------------------------------------------------------------------------



# Create COUSTOM Pytorch DATASET with features and labels----------------------------------------------------------------------------
# Source: [21] https://stackoverflow.com/questions/56774582/adding-custom-labels-to-pytorch-dataloader-dataset-does-not-work-for-custom-data



class Dataset:
    def __init__(self, root):
        """Init function should not do any heavy lifting, but
            must initialize how many items are availabel in this data set.
        """
        from os import listdir
        from os.path import isfile, join
        self.featurepath = root + "/features"
        self.labelpath = root + "/labels"

        self.ROOT = root
        self.featurelist = [f for f in listdir(self.featurepath) if isfile(join(self.featurepath, f))]
        self.labellist = [f for f in listdir(self.labelpath) if isfile(join(self.labelpath, f))]

    def __len__(self):
        """return number of points in our dataset"""
        return len(self.featurelist)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        imagepath =   self.featurepath+"/"+ "Feature" +str(idx)+".npy"
        img = np.load(imagepath)
        labelpath =   self.labelpath+"/"+ "label"+str(idx)
        #Below is to read and retrieve its contents, rb-read binary
        with open(labelpath, "rb") as f:
            label = pickle.load(f) 
            labels_2 = np.array(label[0])
            labels_4 = np.array(label[1])
            labels_8 = np.array(label[2])
            labels_16 = np.array(label[3])
        return img, labels_2 , labels_4 , labels_8, labels_16




def define_dataset(self,train_test_switch):

    dataset = None
    try:
        if train_test_switch == "train":
            trainDatasetSaveplace = FileParentPath + "/Datasets/train"
            trainDataset = self.Dataset(trainDatasetSaveplace)
            #Now, you can instantiate the DataLoader:
            trainDataloader = DataLoader(trainDataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
            dataiter = iter(trainDataloader)
            trainDataset = dataiter.next()
            trainDataset  = transforms.ToTensor()
            dataset = trainDataset
            dataloader = trainDataloader
        
        elif train_test_switch == "test":
            #DEFINING TEST DATA LOADER FOR TESTINGs

            testDatasetSaveplace = FileParentPath + "/Datasets/test"
            testDataset = self.Dataset(testDatasetSaveplace)
            testDataLoader = DataLoader(testDataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True )
            #This will create batches of your data that you can access as:
            testiter=  iter(testDataLoader)
            testDataset = testiter.next()
            testDataset  = transforms.ToTensor()
            dataset = testDataset
            dataloader = testDataLoader



    except:
        PrintException()
        print("Did you rebuild the train test data? Please check")
        input("")
        pass

    #####################################################################################################################
    #       DATA COMPLETE       ->     NOW MACHINE LEARNING PART
    assert dataset is not None
    #else: input("Dataset is None")

    return dataset, dataloader

'''    