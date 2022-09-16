# -*- coding: utf-8 -*-
#TODO: NEW MAIN MODE: IMPLEMENT RL MODE FOR DEPLOYING AE to TASK using min, max, map search fractal value for loss function or custom loss function or target mapping (draw function like fan curve)
#      DataExplorer: Streamlit dashboard for controlling in and output, & config.py 
#      MPM PACKER: Map traditional scanning order to mpm to aquire samples from scanning probes (like laser scanning microscope)
# !!!! Latent space Explortion mode: passing "few" labelt batches through the encoder and map the gaussian distribution to the desired label + vector additon/subtraction: Like... Queen = King - man + Woman
# !!!! Discriminator tells the liklyhood -> maybe heatmap on latent space.... -> how to sample standard deviation?
'''
NOTE FÜRS MEETING
Wie groß ist die adventure stelle und kann nick da von was abhaben fürs einlernen?
Vlcht mal alike vorstellen, was ich in freizeit und foe entwickelt habe. maybe mal jupyter notebook oder das masking feature vorstellen für parameter&performance estimation

1 mal im monat mw ok? rest via Server + meeting
fürs institut wenn nix zu tun ein dashboard oder videos für Lasertechnik & selfpromotion
Vlcht möglichkeit der software entwicklung, dass Alike zur Parametervorhersage eignen lässt durch masking etc
Lizenzgeber LHM + Me




'''


MainFunctionDiscription = " Type: boxcounting(bc),  mkDataset(mkdata), ConvNet_BoxCounting(cnn_bc), DataExplorer(da) , or FractalGAN(gan) : "
import time

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("MainFunction", help=" Type: boxcounting(bc),  mkDataset(mkdata), ConvNet_BoxCounting(cnn_bc), DataExplorer(da) , or FractalGAN(gan) ; for what you wanna do",type=str)

parser.add_argument("--MainFunction", type=str, default="NOT any", help="number of epochs of training")
parser.add_argument("--device", type=str, default="gpu", help="Set device to cpu/gpu  default is gpu")

parser.add_argument("--n_epochs", type=int, default=5, help="Type: boxcounting(bc),  mkDataset(mkdata), ConvNet_BoxCounting(cnn_bc), DataExplorer(da) , or FractalGAN(gan) :")
#parser.add_argument("--hyperopt", type=str, default="on", help="type --hyperopt=off to use the given/default arguments instead of Hyperparameteroptimization")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the 2d latent code like AxB, 8x0 by default 0 has to be calced into the right format")
parser.add_argument("--img_size", type=tuple, default=(32,32), help="size of each image dimension in (y,x), HAVE TO BE DEVISABLE BY 2... 2,4,8,16,18,...212..512")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10000000, help="interval between image sampling")
parser.add_argument("--autoencoder", type=str, default="off", help="type --autoencoder=on to use just the encoder/decoder without the discriminator")
parser.add_argument("--verbosity", type=bool, default=False, help="Set verbosity True/False for display results, or show additional data for debugging")

opt, unknown = parser.parse_known_args()
print(f"Unknown args: {unknown}")
#TODO: CHECK IF opt.img_size in[f for f in x^n] or check if devisiable/2==True has to be the correct thing
opt = parser.parse_args()
#opt.device = "cpu"
#opt.img_size= (256,256)
opt.SaveEveryModel = False
'''
set_img_size = input("Whats the image size?")  # TODO: Redo with x and y tuples
try: 
    int(set_img_size)
except Exception as e:
    print(e)    
    set_img_size = 0

if set_img_size is 0:
    pass
else:
    #opt.img_size= (32,32)
    opt.img_size= (set_img_size,set_img_size)
'''
#opt.n_cpu = 24

opt.residual = True
opt.hyperopt = "on"
opt.Population_Init_Threshold = 1000
opt.Generation_limit = 10

opt.autoencoder = "on"      #train/test without discriminator
#opt.autoencoder = "off"    #train/test with discriminator network

opt.Max_parallel_layers = 3
opt.Max_Lenght = 6

#print(opt)
opt.No_latent_spaces = 1
opt.UpdateMaskEvery = 10


#### INPAINTING OPTIONS
opt.superres = False


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

### Adapting image size and latent dim

#opt.latent_dim_x = int((opt.img_size[1]/ opt.img_size[0]) * opt.latent_dim)
#print("opt.latent_dim_x is  ",opt.latent_dim_x)
#opt.latent_dim_yx = (opt.latent_dim[0], latent_dim_x)
#time.sleep(2)

opt.img_shape = (opt.channels, opt.img_size[0], opt.img_size[1])

'''
#TODO: CHECK under WINDOWS IF DEPRECHEATED
#Cause on Windows pytorch dataloader doesn't support multicore dataloader
import platform
Operating_system = platform.system()
#print(Operating_system)

if Operating_system == 'Windows':
    print("Windows detected, dataloader multiprocessing not avaiable. Set n_cpu to 0 ")
    opt.n_cpu = 0
'''
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials  #hyperoptimization libary


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

if opt.device == "cpu":
    print("CPU USED FOR NN")
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" #To disable GPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
elif opt.device == "cuda":
    pass
elif opt.device == "cuda0":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
elif opt.device == "cuda1":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"  
elif opt.device == "cuda2":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"  






MainFunction = opt.MainFunction
#print("This script is running in ",MainFunction," Mode!")


from pathlib import Path
#HOME DIRECTORY
home = str(Path.home())
#For the directory of the script being run:
FileParentPath = str(Path(__file__).parent.absolute())
opt.FileParentPath = FileParentPath


#Import own scripts
import Loader       #loads the data containing in a chosen folder and returns it to the MPMPacker
#import BoxcountFeatureExtr_v2 # Calculates the Boxcount structure map and appends it to the Data as a Lable so later the gpu version can calc and gradient decend
import CNN_Boxcount_encoder_v2 as CNN_Boxcount_encoder
import FractalGANv13 as FractalGAN

opt.n_cpu = 12

opt.device = FractalGAN.get_device()

#Import tkinter to be able to use choose file/directory promt in windowed mode
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askdirectory

import numpy as np
import pickle
import multiprocessing 
import torch                                    #Pytorch machine learning framework.
import itertools
print("IMPORTS DONE")

###########################FUNCTIONS
def Box_Counting(opt):
    if opt.SingleDataORDatafolder.lower() == "d"  or opt.SingleDataORDatafolder =="":
        print("Please choose the Folder you want the program work with")
        opt.DataFolder = askdirectory()
        DataHandler = Loader.DataHandler(opt)
        DataHandler = Loader.choose_DataFormat(DataHandler)
        DataHandler = Loader.IterateOverDataFolder(DataHandler)

    elif opt.SingleDataORDatafolder.lower() == "s":
        print("Please choose the file you want to characterize")
        root = Tk()
        root.filename =  filedialog.askopenfilename(initialdir = home ,title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        print(root.filename)
        opt.ChosenFilePath = root.filename
        FileExtension, MIME = Loader.guessFiletype(opt.ChosenFilePath)
        print('File extension: %s' % FileExtension)
        print('File MIME type: %s' % MIME)
        #print(root.Filename)
        filespecs = str(opt.ChosenFilePath), FileExtension, MIME

        DataHandler = Loader.DataHandler(opt)
        #DataHandler = Loader.choose_DataFormat(DataHandler)
        DataHandler.maxvalue = 256.0
        DataTypeChoice = 2  # for pictures 2
        DataHandler.chosenDataFormat = DataHandler.DataFormatlist[DataTypeChoice]
        npOutputFile, codierung = Loader.LoadArbitraryFileAs_npArray(DataHandler,filespecs)
        print("Begin box counting characterizement")

        print("File loaded...loaded shape is", npOutputFile.shape, "  and coding", codierung)

        print("For spacial Boxcounting just concatenate channels side by side... delete this here in iterateoverfoler(), when working")

        for index, channel in enumerate(codierung[-1]):
            if index == 0:
                output = npOutputFile[:,:,index]
            else:
                output = np.concatenate((output, npOutputFile[:,:,index]),axis=1)

        npOutputFile = output

        #BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.SinglethreadBoxcount(npOutputFile,DataHandler.maxvalue)
        BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadChoosingBoxcount(npOutputFile,DataHandler)

        if opt.Show_results.lower() == "y" or opt.SaveOrNotToSave.lower() == "y":
            import DataExplorer
            Visualizer = DataExplorer.Visualize()
            DoYouWantToSave = opt.SaveOrNotToSave

            if opt.WhichBoxCounting.lower() == "s":
                #if spacial boxcounting, then display images
                Visualizer.characterize_Image_and_plot(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict,  DoYouWantToSave, opt.Show_results.lower() )
            elif opt.WhichBoxCounting.lower() == "b":
                print("If original Boxcount is chosen, there are just the counts and lacs... no pics")
                print("Printing Boxcounts/Lacs")
                Visualizer.Print_characterized_image(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict, DoYouWantToSave)

        print("Characterizing done")
    print("ID of Box_Counting: {}".format(os.getpid())) 


def Create_Dataset_Worker(opt,DataHandler):
    DataHandler = Loader.make_train_test_Datasets_multicore(DataHandler,opt)    #AND CALCS BOXCOUNTS on the run
    #DataHandler = Loader.make_train_test_Datasets(DataHandler,opt)    #AND CALCS BOXCOUNTS on the run
    #print(DataHandler.Dataset)
    print(f'{len(DataHandler.Dataset)} Entrys are in generated Dataset')
    print("Dataset created!")
    print("ID of Create_Dataset_Worker: {}".format(os.getpid())) 


def CNN_Spacial_Boxcount_Worker(TrainTestPredict):
    print("Convolutional Neural Network for spacial Box Counting is activated")
    print("Do you want to train a Model, test a Model, or predict with the model")                
    
    device = FractalGAN.get_device()
    print("Chosen Device is",device)
    
    ModelnameList = None
    opt.verbosity = False

    BoxCountEnc_model= CNN_Boxcount_encoder.CNN_BC_enc(opt)
    DataHandler = Loader.DataHandler(opt)
    if TrainTestPredict.lower() == "train":
        print("define train dataset")
        train_test_switch = "train"
        trainDataset, trainDataloader = DataHandler.define_dataset(train_test_switch, opt.ProjectName)
        BoxCountEnc_model.begin_training(trainDataset, trainDataloader)

    elif TrainTestPredict.lower() == "test":
        print("Define test dataset")
        train_test_switch = "test"
        testDataset, testDataLoader = DataHandler.define_dataset(train_test_switch, opt.ProjectName)
        BoxCountEnc_model.validation(testDataset, testDataLoader)
        print("Validation completed")

    elif TrainTestPredict.lower() == "predict":
        print("Choose the Model for predicting spacial Boxcounting")
        root = Tk()
        root.filename =  filedialog.askopenfilename(initialdir = FileParentPath + "/models/SpacialBoxcountModels/" ,title = "Select file",filetypes = (("model files","*.model"),("all files","*.*")))
        print(root.filename)
        Modelname = root.filename[:-6]
        __, __, __, __, opt.batch_size, __, __, __, __, __  = Modelname.split("_") 
        opt.batch_size = int(opt.batch_size)
        print("LOADING BoxcountEncoder Model named: \n" ,Modelname )

        NetParametersSaveplace = Modelname +".netparams"
        BoxCountNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
        BoxcountEncoder = CNN_Boxcount_encoder.BoxCountEncoder(BoxCountNetParameters)

        Modelsaveplace = Modelname +".model"
        try:
            BoxcountEncoder.load_state_dict(torch.load(Modelsaveplace, map_location=device))
        except:
            BoxcountEncoder.load_state_dict(torch.jit.load(Modelsaveplace, map_location=device))


        print("BoxcountEncoder generated")
        train_test_switch = "test"
        testDataset, testDataLoader = DataHandler.define_dataset(train_test_switch,opt.ProjectName)

        #testDataset, testDataLoader = BoxCountEnc_model.define_dataset("test")
        print("Dataloader/Dataset defined")
        
        for  i, BatchToBePredicted  in enumerate(testDataLoader):
            display = True
            BCR_LAK_map_2, BCR_LAK_map_4, BCR_LAK_map_8, BCR_LAK_map_16 = BoxCountEnc_model.predict(BoxcountEncoder, device, BatchToBePredicted, display)
            
            print("single batch predicted")
            #input("Press for next batch")
        

def FractalGAN_Worker(opt):
    global all_ness_params
    print("Placeholder for THE GAN submodule")
    #import Loader       #loads the dataset
    device = FractalGAN.get_device()
    print("Chosen Device is",device)
    #device = "cpu"
    ########################################################################################
    #####################                   CNN_BoxCount               #####################
    ########################################################################################
    print("Convolutional Neural Network for spacial Box Counting is initilized")
    ModelnameList = None
    opt.verbosity = False

    #INIT#################################################
    DataHandler = Loader.DataHandler(opt)

    opt.Mode = 2
    if opt.Mode == 1:
        BoxCountEnc_model= CNN_Boxcount_encoder.CNN_BC_enc(opt)
        #newtrained model
        #Modelname = "Loss0.529---n_epochs_90_batch-size_512_learning-rate_0.063_beta-decay_0.928_0.664"
        #Modelname = "Loss0.017---n_epochs_115_batch-size_8_learning-rate_0.004_beta-decay_0.171_0.104"
        #Modelname = "Loss0.019---n_epochs_70_batch-size_2_learning-rate_0.007_beta-decay_0.391_0.487"
        #Modelname = "Loss0.014---n_epochs_80_batch-size_128_learning-rate_0.0_beta-decay_0.079_0.837"
        print("Choose the Model for flavoring GAN")
        root = Tk()
        root.filename =  filedialog.askopenfilename(initialdir = FileParentPath + "/models/SpacialBoxcountModels/" ,title = "Select file",filetypes = (("model files","*.model"),("all files","*.*")))
        print(root.filename)
        Modelname = root.filename[:-6]
        print("LOADING BoxcountEncoder Model named: \n" ,Modelname )
        NetParametersSaveplace =FileParentPath+ "/models/SpacialBoxcountModels/"+ Modelname +".netparams"
        BoxCountNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
        #INIT BC ENC NETWORK STRUCTURE
        BoxcountEncoder = CNN_Boxcount_encoder.BoxCountEncoder(BoxCountNetParameters)

        Modelsaveplace = Modelname +".model"
        #LOAD WEIGHS & BIASES
        try:
            BoxcountEncoder.load_state_dict(torch.load(Modelsaveplace, map_location=device))
        except:
            BoxcountEncoder.load_state_dict(torch.jit.load(Modelsaveplace, map_location=device))
        print("BoxcountEncoder generated")

    elif opt.Mode ==2:
        print("CPU Mode chosen for Boxcount Encoding")
        BoxCountEnc_model = None
        BoxcountEncoder = None

    #TRAINTESTORVALIDATE

    if opt.TrainOrTest.lower() ==  "train":
        if opt.hyperopt == "on":
            #ContainedLatentDims = [2,4,8,16,32,64]
            ContainedLatentDims = [2]
            while True:
                ContainedLatentDims.append(ContainedLatentDims[0]*ContainedLatentDims[-1])
                if ContainedLatentDims[-1] >= opt.img_size[0]:
                    break

            HyperParameterspace = {
                'n_epochs':hp.choice('opt.n_epochs', range(1,3) ),
                'lr':hp.uniform('opt.lr', 0.0001 , 0.01 ), 
                'b1':hp.uniform('opt.b1', 0.8 , 1.0 ),
                'b2':hp.uniform('opt.b2', 0.8 , 1.0 ),
                'latent_dim':hp.choice('opt.latent_dim', ContainedLatentDims ),
            }     

        elif opt.hyperopt == "off":
            HyperParameterspace = {
                'n_epochs':opt.n_epochs,
                'lr':opt.lr, 
                'b1':opt.b1,
                'b2':opt.b2,
                'latent_dim':opt.latent_dim,
            }    
    

        train_test_switch = opt.TrainOrTest.lower()
        DataHandler = Loader.DataHandler(opt)
        Dataset, DataLoader = DataHandler.define_dataset(train_test_switch,opt.ProjectName)
        previous_Best_Loss = None
        opt.first_time = True


        all_ness_params = opt, Dataset, DataLoader, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace
        happend = 0
        '''
        while True:
            try:
                FractalGAN.begin_training(all_ness_params)
            except:
                PrintException()
                print("Sleeping 1m")
                time.sleep(60)
                continue
        '''
        FractalGAN.begin_training(all_ness_params)


    elif opt.TrainOrTest.lower() == "val":
        import pathlib
        path = str(pathlib.Path(__file__).parent.absolute())   
        if sys.platform == "linux" or sys.platform == "darwin":
            path = path +"/models/GAN/" + opt.ProjectName+"/"
        elif sys.platform == "win32":
            path = path +"\\models\\GAN\\" + opt.ProjectName+"\\"
        Enc_netparams_file_list = []
        Dec_netparams_file_list = []
        Dis_netparams_file_list = []
        enc_dec_loss_list = []
        dis_loss_list = []

        encoder_dict = {}
        decoder_dict ={}
        discriminator_dict = {}
        unixtime_dict = {}

        for index, FILE in enumerate(os.listdir(path)):
            try:
                    
                if FILE[-6:] == ".model":
                    print(FILE)

                    try:
                        unixtime, lossvalue, Network_type = FILE.split("_")
                    except:
                        lossvalue, Network_type = FILE.split("_")

                    lossvalue  = lossvalue.replace("Loss","")
                    lossvalue  = lossvalue.replace("---","")
                    lossvalue = float(lossvalue)
                    Network_type = Network_type.replace(".model","")

                    with open(path+FILE[:-6]+".netparams", "rb") as f:
                        NetParameters = pickle.load(f)
                        #print("Extracted NETPARAMETERS ARE ",NetParameters)
                    print("chosen unixtime", unixtime)
                    unixtime_dict[unixtime] = lossvalue  


                    if Network_type == "ENCODER":
                        encoder_dict[unixtime] = [FILE,NetParameters,lossvalue]
        
                    elif Network_type == "DECODER":
                        decoder_dict[unixtime] = [FILE,NetParameters,lossvalue]
                    
                    elif Network_type == "DISCRIMINATOR":
                        discriminator_dict[unixtime] = [FILE,NetParameters,lossvalue]
            except:
                PrintException()
                pass

        '''
        # JUST TO CREATE A CSV WITH UNIXTIMEDICT AND CORRESPONDING LOSSVALUE
        '''
        import csv
        now = int(time.time())
        #my_dictionary = {'values': 678, 'values2': 167, 'values6': 998}
        input("Try to write dictionary")
        with open(f'VarAE loss over time_{now}.csv', 'w') as f:
            for idxs, key in enumerate(unixtime_dict.keys()):
                f.write("%s, %s, %s\n" % (idxs, key, unixtime_dict[key]))
        input("Dict written... please check")

        ###################################################################################
        # File list created.... now validation begins
        ###################################################################################
        print("Model list created now validation")
        print(unixtime_dict)

        train_test_switch = "test"
        DataHandler = Loader.DataHandler(opt)
        Dataset, DataLoader = DataHandler.define_dataset(train_test_switch,opt.ProjectName)
        #device = FractalGAN.get_device()
        device = 'cpu'
        #sorteddict = sorted(unixtime_dict.items(), key = lambda x:x[-1]) #sort the retreived models by their loss value [1][-1] = values, last value in list
        #sorteddict = dict(sorted(unixtime_dict.items(), key = lambda x:x, reverse = True)) #sort the retreived models by their loss value [1][-1] = values, last value in list
        import operator
        #sort the received models by increasing loss value, so best comes first
        sorteddict = dict(sorted(unixtime_dict.items(), key = operator.itemgetter(1)))

        for timestamp, value in sorteddict.items():
            try:
                print("Now unixtime-timestamp  "+ str(timestamp)+"  is used")
                print(f"Value of sorteddict is {value}")
                File , NetParameters, lossvalue = encoder_dict[timestamp]
                print(f"Loss is {lossvalue} and should be sorted from 0 to 1")
                encoder_path = path + File

                File , NetParameters, lossvalue = decoder_dict[timestamp]
                decoder_path = path + File

                Encodername = encoder_path[:-6]
                Decodername = decoder_path[:-6]

                if opt.autoencoder == "off":    
                    File , NetParameters, lossvalue = discriminator_dict[timestamp]
                    discriminator_path = path + File
                    Discriminatorname = discriminator_path[:-6]
                    #Modelname, EncDeDis = Modelname.split("_")

                ####################################
                #load ENCODER
                ####################################

                NetParametersSaveplace = Encodername +".netparams"
                EncoderNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
                EncoderNetParameters['device'] = device
                print("EncoderNetParameters",EncoderNetParameters)
                opt = EncoderNetParameters['opt']
                
                print("opt.latent", opt.latent_dim)

                #saveplace = Encodername +".model"
                saveplace = encoder_path
                encoder = FractalGAN.Encoder(EncoderNetParameters)

                try:
                    encoder.load_state_dict(torch.load(saveplace, map_location=device))
                except:
                    encoder.load_state_dict(torch.jit.load(saveplace, map_location=device))

                encoder.eval()   #to disable backpropagation, so don't adjust any weights and biases
                FractalGAN.count_parameters(encoder)


                ####################################
                #load DECODER
                ####################################
                NetParametersSaveplace =Decodername +".netparams"
                DecoderNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
                DecoderNetParameters['device'] = device
                #saveplace = Decodername +".model"
                saveplace = decoder_path
                decoder = FractalGAN.Decoder(DecoderNetParameters)

                try:
                    decoder.load_state_dict(torch.load(saveplace, map_location=device))
                except:
                    decoder.load_state_dict(torch.jit.load(saveplace, map_location=device))
                
                FractalGAN.count_parameters(decoder)

                decoder.eval()   #to disable backpropagation, so don't adjust any weights and biases


                if opt.autoencoder == "off":
                        
                    ####################################
                    #load Discriminator
                    ####################################
                    NetParametersSaveplace =Discriminatorname +".netparams"
                    DiscriminatorNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
                    DiscriminatorNetParameters['device'] = device
                    saveplace = Discriminatorname +".model"

                    discriminator = FractalGAN.Discriminator(DiscriminatorNetParameters)

                    try:
                        discriminator.load_state_dict(torch.load(saveplace, map_location=device))
                    except:
                        discriminator.load_state_dict(torch.jit.load(saveplace, map_location=device))
                    
                    FractalGAN.count_parameters(discriminator)

                    discriminator.eval()   #to disable backpropagation, so don't adjust any weights and biases

                else:
                    discriminator = None
                    DiscriminatorNetParameters = None


                previous_Best_Loss = None
                HyperParameterspace = None

                ########################################################################################
                #####################                   Inpainting Layers          #####################
                ########################################################################################  
                print("Generating Masking Layers")
                print("image size is", opt.img_size)
                    
                opt.noisebool = True
                #opt.std = 0.001        #light disturbance
                opt.std = 0.01      #moderate Disurbance
                #opt.std = 0.1      #hard Disturbance

                opt.std_decay_rate = 0 

                # if maskbool is None, Random masking is applied
                #opt.maskbool = None
                opt.maskbool = False
                #maskmean       x                       Y
                opt.maskmean = opt.img_size[1]/2 , opt.img_size[0]/2    #just the center for exploring  
                #               =          x                         Y                    
                opt.maskdimension =  int(opt.img_size[1]/8) , int(opt.img_size[0]/8)

                opt.LetterboxBool = False
                opt.LetterboxHeight = 30

                opt.PillarboxBool = False
                opt.PillarboxWidth = 10
                opt.device = 'cpu'
                opt.InpaintingParameters = {
                    'opt': opt,
                    'superresolution': (opt.superres, 2),
                    'noise': (opt.noisebool, opt.std, opt.std_decay_rate),
                    'mask': (opt.maskbool, opt.maskmean , opt.maskdimension),
                    'Letterbox': (opt.LetterboxBool, opt.LetterboxHeight),
                    'Pillarbox': (opt.PillarboxBool, opt.PillarboxWidth),

                }

                #inpainting = FractalGAN.Inpainting(opt.InpaintingParameters)

                inpainting = None

                #opt.autoencoder = autoencoder

                all_ness_params = opt, Dataset, DataLoader, inpainting, opt.InpaintingParameters ,encoder,EncoderNetParameters,  decoder, DecoderNetParameters,  discriminator , DiscriminatorNetParameters, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace 
                
                FractalGAN.TestGAN(all_ness_params)
                
                delete = input("Do you want to delete the current model or just continue to next model? (y/N)")
                if delete.lower() == "y":
                    try:
                        os.remove(encoder_path)
                    except:
                        PrintException()
                        pass
                    try:
                        os.remove(encoder_path[:-6]+".netparams")
                    except:
                        PrintException()
                        pass                    
                    try:
                        os.remove(decoder_path)
                    except:
                        PrintException()
                        pass                    
                    try:
                        os.remove(decoder_path[:-6]+".netparams")
                    except:
                        PrintException()
                        pass

                    if opt.autoencoder == "off":  
                        try:    
                            os.remove(discriminator_path)
                            os.remove(discriminator_path[:-6]+".netparams")                
                        except: 
                            PrintException()
                            pass
                            
                elif delete.lower() == "n":
                    continue
            
            except:
                print("Some error happend! What do you want to do?")
                PrintException()
                delete = input("Do you want to delete the model (y/N)")
                try:
                    if delete.lower() == "y":
                        try:
                            os.remove(encoder_path)
                        except:
                            PrintException()
                            pass
                        try:
                            os.remove(encoder_path[:-6]+".netparams")
                        except:
                            PrintException()
                            pass                    
                        try:
                            os.remove(decoder_path)
                        except:
                            PrintException()
                            pass                    
                        try:
                            os.remove(decoder_path[:-6]+".netparams")
                        except:
                            PrintException()
                            pass

                        if opt.autoencoder == "off":    
                            os.remove(discriminator_path)
                            os.remove(discriminator_path[:-6]+".netparams")
                    continue
                except:
                    PrintException()
                    pass

    if opt.TrainOrTest.lower() ==  "test":
        train_test_switch = opt.TrainOrTest.lower()
        DataHandler = Loader.DataHandler(opt)
        Dataset, DataLoader = DataHandler.define_dataset(train_test_switch,opt.ProjectName)

        '''
        #LOADING CHOSEN MODEL
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
        '''

        #device = FractalGAN.get_device()
        device = "cpu"
        
        
        print("Choose the Model for loading and testing already trained GAN")
        root = Tk()

        if sys.platform == "linux" or sys.platform == "darwin":
            initialdir = FileParentPath + "/models/GAN/" + opt.ProjectName + "/"
        elif sys.platform == "win32":
            initialdir = FileParentPath + "\\models\\GAN\\" + opt.ProjectName + "\\"

        root.filename =  filedialog.askopenfilename(initialdir = initialdir ,title = "Select file",filetypes = (("model files","*.model"),("all files","*.*")))
        print(root.filename)
        Modelname = root.filename[:-6]
        Unixtime, Modelname, EncDeDis = Modelname.split("_")

        Encodername = Unixtime +"_"+ Modelname+"_ENCODER"
        Decodername = Unixtime +"_"+Modelname + "_DECODER"
        '''
        print("Choose the CORRESPONDING Model for loading the DISCRIMINATOR")
        root = Tk()
        root.filename =  filedialog.askopenfilename(initialdir = initialdir ,title = "Select file",filetypes = (("model files","*.model"),("all files","*.*")))
        print(root.filename)
        Discriminatorname = root.filename[:-6]
        #Modelname, EncDeDis = Modelname.split("_")
        '''


        ####################################
        #load ENCODER
        ####################################
        NetParametersSaveplace = Encodername +".netparams"

        EncoderNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
        EncoderNetParameters['device'] = device
        print("EncoderNetParameters",EncoderNetParameters)

        opt = EncoderNetParameters['opt']
        print("opt.latent", opt.latent_dim)
        saveplace = Encodername +".model"

        encoder = FractalGAN.Encoder(EncoderNetParameters)

        try:
            encoder.load_state_dict(torch.load(saveplace, map_location=device))
        except:
            encoder.load_state_dict(torch.jit.load(saveplace, map_location=device))

        encoder.eval()   #to disable backpropagation, so don't adjust any weights and biases



        ####################################
        #load DECODER
        ####################################
        NetParametersSaveplace =Decodername +".netparams"
        DecoderNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
        DecoderNetParameters['device'] = device
        saveplace = Decodername +".model"

        decoder = FractalGAN.Decoder(DecoderNetParameters)

        try:
            decoder.load_state_dict(torch.load(saveplace, map_location=device))
        except:
            decoder.load_state_dict(torch.jit.load(saveplace, map_location=device))

        decoder.eval()   #to disable backpropagation, so don't adjust any weights and biases



        if opt.autoencoder == "off":
            print("Choose the CORRESPONDING Model for loading the DISCRIMINATOR")
            root = Tk()
            root.filename =  filedialog.askopenfilename(initialdir = initialdir ,title = "Select file",filetypes = (("model files","*.model"),("all files","*.*")))
            print(root.filename)
            Discriminatorname = root.filename[:-6]
            #Modelname, EncDeDis = Modelname.split("_")

            ####################################
            #load Discriminator
            ####################################
            NetParametersSaveplace =Discriminatorname +".netparams"
            DiscriminatorNetParameters = pickle.load(open(NetParametersSaveplace, "rb"))
            DiscriminatorNetParameters['device'] = device
            #DiscriminatorNetParameters['opt'] = opt

            saveplace = Discriminatorname +".model"

            discriminator = FractalGAN.Discriminator(DiscriminatorNetParameters)

            try:
                discriminator.load_state_dict(torch.load(saveplace, map_location=device))
            except:
                discriminator.load_state_dict(torch.jit.load(saveplace, map_location=device))

            discriminator.eval()   #to disable backpropagation, so don't adjust any weights and biases
        
        else:
            DiscriminatorNetParameters = None
            discriminator = None



        previous_Best_Loss = None
        HyperParameterspace = None



        ########################################################################################
        #####################                   Inpainting Layers          #####################
        ########################################################################################  
        print("Generating Masking Layers")
        print("image size is", opt.img_size)
            
        opt.noisebool = True
        #opt.std = 0.001        #light disturbance
        opt.std = 0.02      #moderate Disurbance
        #opt.std = 0.1      #hard Disturbance

        opt.std_decay_rate = 0 

        # if maskbool is None, Random masking is applied
        #opt.maskbool = None
        opt.maskbool = False
        #maskmean       x                       Y
        opt.maskmean = opt.img_size[1]/2 , opt.img_size[0]/2    #just the center for exploring  
        #               =          x                         Y                    
        opt.maskdimension =  int(opt.img_size[1]/8) , int(opt.img_size[0]/8)

        opt.LetterboxBool = False
        opt.LetterboxHeight = 30

        opt.PillarboxBool = False
        opt.PillarboxWidth = 10
        #opt.device = 'cpu'
        InpaintingParameters = {
            'opt': opt,
            'superresolution': (opt.superres, 2),
            'noise': (opt.noisebool, opt.std, opt.std_decay_rate),
            'mask': (opt.maskbool, opt.maskmean , opt.maskdimension),
            'Letterbox': (opt.LetterboxBool, opt.LetterboxHeight),
            'Pillarbox': (opt.PillarboxBool, opt.PillarboxWidth),

        }


        inpainting = None
        opt.InpaintingParameters = None
        #opt.autoencoder = autoencoder
        opt.device = device

        all_ness_params = opt, Dataset, DataLoader, inpainting, InpaintingParameters ,encoder,EncoderNetParameters,  decoder, DecoderNetParameters,  discriminator , DiscriminatorNetParameters, BoxCountEnc_model,BoxcountEncoder,device, previous_Best_Loss  , HyperParameterspace 

        FractalGAN.TestGAN(all_ness_params)





#Main program
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    print("ID of main process: {}".format(os.getpid())) 

    print("Welcome to Alike.  \nThe software to characterize, categorize arbitrary Data! \nSearch, sort & compare text, pictures, Videos, or any kind of Data to another: ")

    print("The Main Functions are...: \n ")
    MainFunction = input(MainFunctionDiscription)
        
    if MainFunction.lower() == "dataexplorer" or MainFunction.lower() == "da":
        print("STARTING dataexplorer !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        ''' 
        TODO: 

        -  BCR LAK NORMALISIEREN, da nahe0 und evtl loss ändern auf absolut/ falls sich +- ausgleichen


        - NETWORK GENERATION INTO FRACtAL GAN, because the network is just continues training, without new random/parent generation

        - Predict loop with sample images for gan

        - insert other picture -> bcr/lac  and combine with diffrent latent representation

        - GENERATOR/DECODER instead of Residual adding the bcr/lak just the loss has to be taken from these layers
        , BECAUSE ITS easyer to generate the pic from bcr/lak in the intercept layers
        , than going through the encoder -> thats why maybe the discriminator doesnt get better, 
        when no meaningfull data is compressed  -> path of least resistance


        Dataexplorer bedient sich des visualizers

        Aufgaben:
        -Bestimme Fractale Dimension -> maybe moeglichkeit der oertlichen dimension (peters dimension) nutze alle boxsizes to 1x1 from multichannel/boxsizes to 1 channel for spacial peters dimension

        -fractal general/spacial search
        -lac edge.detect/Anomaly search/
        -FractalGAN Manipulator:
            - DIscriminator: is this real or fake -> Data detection
            - Pass through: reconstruct data, with dropout/denoise/
            - pass through: define edgecases/reine zustaende and categorize/cluster the latent space by neares neighbor
            - pass through: Generate new things/pictures/data with defined old data
        
        -Search spaces 
            - search in folder/file
            - search in features/labels spacial boxcounting
            -maybe search just the generated/labels for speedup and scaling/zoom
                -#TODO: not just gridwise chunking, but pointer wise with obj detection boxes or random/chosen slices
        
        -      
        
        '''
        pass

    if MainFunction == "bc" or MainFunction == "boxcounting":
        #print("STARTING boxcounting !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Box Counting chosen")
        print("Do you want to execute... \n...standard box counting? (b) \n...spacial box counting? (s)")
        opt.WhichBoxCounting = input('Please choose: (S/b): ')
        opt.Show_results = input("Do you want to show the resulting Boxcounts and lacunaritys? (y/N): ")
        opt.SaveOrNotToSave = input("Do you want to save the computed Boxcounts/Lacunaritys. No just for Debugging/Showing the results (Y/n): ")
        if opt.SaveOrNotToSave.lower() == "y":
            opt.SaveAsNpOrPic = input("Do you want to save the Boxcounts as np.arrays or as picture files (n/P)")
        else:
            opt.SaveAsNpOrPic = "p"
        opt.SingleDataORDatafolder = input("Do you want to want to characterize a single DataFile(s) or a whole directory of files (d)? \nPlease choose now (s/D): ")
    
        if opt.SingleDataORDatafolder.lower() == "d"  or opt.SingleDataORDatafolder =="":
            print("Please choose the Folder you want the program work with")
            opt.DataFolder = askdirectory()
            DataHandler = Loader.DataHandler(opt)
            DataHandler = Loader.choose_DataFormat(DataHandler)
            DataHandler = Loader.IterateOverDataFolder(DataHandler)

        elif opt.SingleDataORDatafolder.lower() == "s":
            print("Please choose the file you want to characterize")


            root = Tk()
            root.filename =  filedialog.askopenfilename(initialdir = FileParentPath ,title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            print(root.filename)
            opt.ChosenFilePath = root.filename
            FileExtension, MIME = Loader.guessFiletype(opt.ChosenFilePath)
            print('File extension: %s' % FileExtension)
            print('File MIME type: %s' % MIME)
            #print(root.Filename)
            filespecs = str(opt.ChosenFilePath), FileExtension, MIME

            DataHandler = Loader.DataHandler(opt)
            #DataHandler = Loader.choose_DataFormat(DataHandler)
            DataHandler.maxvalue = 256.0
            DataTypeChoice = 2  # for pictures 2
            DataHandler.chosenDataFormat = DataHandler.DataFormatlist[DataTypeChoice]
            npOutputFile, codierung = Loader.LoadArbitraryFileAs_npArray(DataHandler,filespecs)
            print("Begin box counting characterizement")

            print("File loaded...loaded shape is", npOutputFile.shape, "  and coding", codierung)

            print("For spacial Boxcounting just concatenate channels side by side")

            for index, channel in enumerate(codierung[-1]):
                if index == 0:
                    output = npOutputFile[:,:,index]
                else:
                    output = np.concatenate((output, npOutputFile[:,:,index]),axis=1)
            print("Concatenated npOutputFile shape is: ", npOutputFile.shape)

            npOutputFile = output

            BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadChoosingBoxcount(npOutputFile,DataHandler)

            if opt.Show_results.lower() == "y" or opt.SaveOrNotToSave.lower() == "y":
                import DataExplorer
                Visualizer = DataExplorer.Visualize()
                DoYouWantToSave = opt.SaveOrNotToSave

                if opt.WhichBoxCounting.lower() == "s":
                    #if spacial boxcounting, then display images
                    Visualizer.characterize_Image_and_plot(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict,  DoYouWantToSave, opt.Show_results.lower() )
                elif opt.WhichBoxCounting.lower() == "b":
                    print("If original Boxcount is chosen, there are just the counts and lacs... no pics")
                    print("Printing Boxcounts/Lacs")
                    Visualizer.Print_characterized_image(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict, DoYouWantToSave)

            else:
                pass
            print("Characterizing done")

            #
            #BoxCountingProcess = multiprocessing.Process(target = Box_Counting,args=(opt,))
            #BoxCountingProcess.start()


    if MainFunction == "mkDataset" or MainFunction.lower() == "mkdata":
        print("STARTING mkDataset !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        opt.ProjectName = input("Whats the name of your project?")
        print("Please choose the Folder you want the program work with")
        opt.DataFolder = askdirectory()
        #This Program aimes to take any given data as an input, pack it onto a Multi-parameter-matrix representation to calc the spacial boxcountdistribution 
        #so characterization/categorization/DataSearch/Datageneration from it is possible
        #opt = Loader.choose_custom_or_mnist_dataset(opt)
        DataHandler = Loader.DataHandler(opt)
        DataHandler = Loader.choose_DataFormat(DataHandler)   
        #DataHandler.precision = input("Set Precision for balancing train/test dataset: 0: No Balancing/50/50split     1...3...5:Coarse Balaning (many train/few test)      5...7...9: Fine balancing (few train/many test)   ")         
        #DataHandler = Loader.make_train_test_Datasets(DataHandler,opt)    #AND CALCS BOXCOUNTS on the run
        DataHandler = Loader.make_train_test_Datasets_multicore(DataHandler,opt)    #AND CALCS BOXCOUNTS on the run
        
        
        #CreateDataSetProcess = multiprocessing.Process(target = Create_Dataset_Worker, args=(opt,DataHandler,))
        #CreateDataSetProcess.start()



    if MainFunction.lower() == "cnn_bc":
        print("STARTING cnn_bc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        TrainTestPredict = input("Type train / test / predict ")

        CreateCNNProcess = multiprocessing.Process(target = CNN_Spacial_Boxcount_Worker,args=(TrainTestPredict,))
        CreateCNNProcess.start()

    if MainFunction.lower() == "FractalGAN" or MainFunction.lower() == "gan":
        print("STARTING FRACTALGAN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        opt.TrainOrTest = input("Do you want to train or test the model(train/test/val): ")
        opt.ProjectName = input("Whats the name of your project?")
        try:
            if sys.platform == "linux" or sys.platform == "darwin":
                modelpath = FileParentPath + "/models/GAN/" + opt.ProjectName + "/"
            elif sys.platform == "win32":
                modelpath = FileParentPath + "\\models\\GAN\\" + opt.ProjectName + "\\"
            os.mkdir(modelpath)
        except:
            PrintException()
            pass

        FractalGAN_Worker(opt)
        #CreateGANProcess = multiprocessing.Process(target = FractalGAN_Worker , args=(opt,))
        #CreateGANProcess.start()


    try:
        BoxCountingProcess.join()
        print("BoxCountingProcess is alive: {}".format(BoxCountingProcess.is_alive()))
    except:
        pass
    
    try:
        CreateDataSetProcess.join()
        print("CreateDataSetProcess is alive: {}".format(CreateDataSetProcess.is_alive()))
    except:
        pass

    try:
        CreateCNNProcess.join()
        print("CreateCNNProcess is alive: {}".format(CreateCNNProcess.is_alive()))
    except:
        pass

    try:
        CreateGANProcess.join()
        print("CreateGANProcesstProcess is alive: {}".format(CreateGANProcess.is_alive()))
    except:
        pass
    
    print("░░░░░█▐▓▓░████▄▄▄█▀▄▓▓▓▌█ Here \n░░░░░▄█▌▀▄▓▓▄▄▄▄▀▀▀▄▓▓▓▓▓▌█ we \n░░░▄█▀▀▄▓█▓▓▓▓▓▓▓▓▓▓▓▓▀░▓▌█ go \n░░█▀▄▓▓▓███▓▓▓███▓▓▓▄░░▄▓▐█▌ again \n░█▌▓▓▓▀▀▓▓▓▓███▓▓▓▓▓▓▓▄▀▓▓▐█ \n▐█▐██▐░▄▓▓▓▓▓▀▄░▀▓▓▓▓▓▓▓▓▓▌█▌ \n█▌███▓▓▓▓▓▓▓▓▐░░▄▓▓███▓▓▓▄▀▐█ \n█▐█▓▀░░▀▓▓▓▓▓▓▓▓▓██████▓▓▓▓▐█ \n▌▓▄▌▀░▀░▐▀█▄▓▓██████████▓▓▓▌█▌\n▌▓▓▓▄▄▀▀▓▓▓▀▓▓▓▓▓▓▓▓█▓█▓█▓▓▌█▌\n█▐▓▓▓▓▓▓▄▄▄▓▓▓▓▓▓█▓█▓█▓█▓▓▓▐\n")
