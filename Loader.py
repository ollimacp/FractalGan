print("Import Loader")
import os

#HOME DIRECTORY
import pathlib
from pathlib import Path
home = str(Path.home())

#For the directory of the script being run:
FileParentPath = str(pathlib.Path(__file__).parent.absolute())

import numpy as np
import pandas as pd
import MpMPacker    #gets the N-Dim Data from the loader and packs every N-Dim datapoint onto a 2D matrix to the feature Extraction
import multiprocessing

from PIL import Image
#import pickle
import filetype
import time
import matplotlib.pyplot as plt
#Audio libary
#from pydub import AudioSegment
#from pydub.playback import play

import BoxcountFeatureExtr_v2 as BoxcountFeatureExtr # Calculates the Boxcount structure map and appends it to the Data as a Lable so later the gpu version can calc and gradient decend


import sklearn.preprocessing as preprocessing
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
import torchvision
Boxsize=[2,4,8,16,32,64,128,256,512,1024]
import pickle
from tkinter.filedialog import askdirectory

#for debugging
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


def showNPArrayAsImage(np2ddArray, title,colormap ):
    plt.figure()
    plt.imshow(np2ddArray,
            interpolation='none',
            cmap = colormap)
    plt.title(title)
    plt.show(block=False)         
    

#just functions via jupyter notebook with !command
def delete_dataset_from_last_time(FileParentPath,ProjectName):
    import shutil

    really = input("-->(y/n):  Do you want to delete the old Dataset or create new dataset? \n BE CARFUL: function can remove whole directorys, so dont change the Fileparentpath")

    if sys.platform == "linux" or sys.platform == "darwin":
        ProjectDir = FileParentPath + "/Datasets/" + ProjectName + "/"
        OldDatasetSaveplace = FileParentPath + "/Datasets/" + ProjectName + "/train/"    
        OldTestDatasetSaveplace = FileParentPath+ "/Datasets/" + ProjectName + "/test/"
    
    elif sys.platform == "win32":
        ProjectDir = FileParentPath + "\\Datasets\\" + ProjectName + "\\"
        OldDatasetSaveplace = FileParentPath + "\\Datasets\\" + ProjectName + "\\train\\"   
        OldTestDatasetSaveplace = FileParentPath+ "\\Datasets\\" + ProjectName + "\\test\\"

    if really =="y":

        try:
            #shutil.rmtree(OldDatasetSaveplace)
            shutil.rmtree(ProjectDir)
            print("Old test dataset deleted!")
        except OSError:
            print("Deleting old test dataset failed, maybe there aren't any... try to create new directories")
            PrintException()

        try:
            os.mkdir(ProjectDir)
            os.mkdir(OldDatasetSaveplace)            
            if sys.platform == "win32":
                os.mkdir(OldDatasetSaveplace+"\\features\\")
                os.mkdir(OldDatasetSaveplace+"\\labels\\")
            else:
                os.mkdir(OldDatasetSaveplace+"/features/")
                os.mkdir(OldDatasetSaveplace+"/labels/")
            print("New dataset dirs created")

        except OSError:
            PrintException()
            print("Creating new directories failed")
            input("Press key to continue")
        #!rm -rf OldDatasetSaveplace
        #!mkdir OldDatasetSaveplace
        #!mkdir OldDatasetSaveplace +"/features/"
        #!mkdir OldDatasetSaveplace +"/labels/"

        '''
        #not needed, because the whole project folder is deleted
        try:
            #shutil.rmtree(OldDatasetSaveplace)
        except OSError:
            print("Deleting old train dataset failed")        #!rm -rf OldDatasetSaveplace
            PrintException()
        ''' 
        try:
            os.mkdir(ProjectDir)
            os.mkdir(OldDatasetSaveplace)            
            if sys.platform == "win32":
                os.mkdir(OldTestDatasetSaveplace+"\\features\\")
                os.mkdir(OldTestDatasetSaveplace+"\\labels\\")
            else:
                os.mkdir(OldTestDatasetSaveplace+"/features/")
                os.mkdir(OldTestDatasetSaveplace+"/labels/")
            print("New dataset dirs created")    
        except OSError:
            PrintException()
            print("Creating new dirs failed")
            input("Press key to continue")
        #!mkdir OldDatasetSaveplace
        #!mkdir OldDatasetSaveplace +"/features/"
        #!mkdir OldDatasetSaveplace +"/labels/"
    else:
        print("Continue without deleting the old dataset")
        print("Try to make folderstructure for project")

        try:
            os.mkdir(ProjectDir)
            os.mkdir(OldDatasetSaveplace)
            if sys.platform == "win32":
                os.mkdir(OldDatasetSaveplace+"\\features\\")
                os.mkdir(OldDatasetSaveplace+"\\labels\\")

            else:   
                os.mkdir(OldDatasetSaveplace+"/features/")
                os.mkdir(OldDatasetSaveplace+"/labels/")
                OldDatasetSaveplace = FileParentPath+ "/Datasets/" + ProjectName + "/test/"

            print("New dataset dirs created")
        except OSError:
            PrintException()
            print("Creating new directories failed")
            #input("Press key to continue")

        try:
            #os.mkdir(ProjectDir)
            os.mkdir(OldTestDatasetSaveplace)
            os.mkdir(OldTestDatasetSaveplace+"/features/")
            os.mkdir(OldTestDatasetSaveplace+"/labels/")        
        except OSError:
            PrintException()
            print("Creating new dirs failed")
            #sinput("Press key to continue")


        input("Press any key to continue")





class DataHandler(object):
    DataFormat = ""
    ChosenDataFileList = ""
    #DataFolder = ""

    def __init__(self,opt):
        self.opt = opt
        try:
            self.DataFolder = opt.DataFolder
        except:
            print("No DataFolder  was chosen")
            self.DataFolder = ""
        #Datalist = [TextFileList, AudiofileList,PictureFileList , VideoFileList, ApplicationFileList, UnknownFileList,TableFileList]
        self.DataFormatlist = ["txt","aud","pic","vid","app","oth","csv"]
        self.max_Value_List = [16.0,100.0,256.0,256.0,16.0,16.0, 2147483646 ]
        self.maxiteration = 3 # cause boxsize 2,4,8,16 = 0,1,2,3 iteration # determines the maximum boxsize
 
    #Source: [20] https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    #@jit(nopython=False)  #,forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
    def pad(self,array, reference, offset):
        """
        array: Array to be padded
        reference: Reference array with the desired shape
        offsets: list of offsets (number of elements must be equal to the dimension of the array)
        """
        # Create an array of zeros with the reference shcdape
        result = np.zeros(reference.shape)
        # Create a list of slices from offset to offset + shape in each dimension
        insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(array.ndim)]
        # Insert the array in the result at the specified offsets
        result[tuple(insertHere)] = array
        return result


    #If a picture/array is more little than the reference shape, than add zeros to the right and bottom with pad()-function to bring it into opt.img_size x opt.img_size
    def reshape_Data(self,PicNumPy,shape, original_shape):
        #if shape is bigger then original, set new max and retry making data
        reshape = (int(original_shape[0]),int(original_shape[1]) )
        print(f"reshaping, cause shape is: {shape}, and wanted original shape {original_shape}")

        if self.opt.verbosity:
            print("reshaping, cause shape and original shape are", shape, original_shape)
            print("reshape",reshape )
            print("PicNumPy shape",PicNumPy.shape)
        ### can add offset; so that pict can be centered
        offset = [0,0,0]
        PicNumPy = self.pad(PicNumPy,np.zeros(reshape),offset )
        return PicNumPy





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
            #if precision is 0, then just 50/50 split dataset into test and train
            if (index % 2) == 0:
                placingcondition = True
                if self.opt.verbosity == True: print("Index is Even -> Train Dataset")

            else:
                placingcondition = False
                if self.opt.verbosity == True: print("Index is Odd -> Test Dataset")


            DensityMap = combinedDensityMap
            DataHandler.lastVariance = np.var(combinedDensityMap)
        else:
            if index <= 256:
                #to populate the field, just add the first 6 elements
                placingcondition = True

                #cause element placingcondition is true, update the variances             
                combinedVariance = np.var(combinedDensityMap)
                DataHandler.lastVariance =  combinedVariance
                print("populate with minimum Pop number", str(index+1) ," of 256")
            else:
                #calculate the Variance from this turn
                combinedVariance = np.var(combinedDensityMap)

                #if the rounded variance of this turn is more or the same of the variance last turn
                if round(combinedVariance,precision) >= round(DataHandler.lastVariance,precision):    
                    placingcondition = True
                    #and update the last variance for next turn with the new value
                    DataHandler.lastVariance =  combinedVariance
                    DensityMap = combinedDensityMap
                else:
                    #dont add this element to the training dataset but to the test dataset
                    placingcondition = False

                if self.opt.verbosity: print("index:", index,"    formerVariance,combinedVariance",round(DataHandler.lastVariance,precision),round(combinedVariance,precision),"so placing is",placingcondition)    
                #input()

        return placingcondition, DensityMap, DataHandler.lastVariance


    '''
    #DEPRECHEATED... or not?
    def create_dataset(self,shape):
        delete_dataset_from_last_time(FileParentPath)
        print("Begin preprocessing train/test datasets")
        self.make_train_test_data(shape)
        print("Datasets created")
        #REBUILDING/Balancing DATA DONE ---------------------------------------------------------------------------------------------
    '''


    # Create COUSTOM Pytorch DATASET with features and labels----------------------------------------------------------------------------
    # Source: [21] https://stackoverflow.com/questions/56774582/adding-custom-labels-to-pytorch-dataloader-dataset-does-not-work-for-custom-data



    class Dataset:
        def __init__(self, root):
            """Init function should not do any heavy lifting, but
                must initialize how many items are availabel in this data set.
            """
            from os import listdir
            from os.path import isfile, join
            #print(sys.platform)
            #print(sys.platform == "win32")
            if sys.platform == "win32":
                self.featurepath = root + "\\features"
                self.labelpath = root + "\\labels"
            else:            
                self.featurepath = root + "/features"
                self.labelpath = root + "/labels"

            self.ROOT = root
            self.featurelist = [f for f in listdir(self.featurepath) if isfile(join(self.featurepath, f))]
            self.labellist = [f for f in listdir(self.labelpath) if isfile(join(self.labelpath, f))]
            print("FeaturePath:",self.featurepath)
            print("LabelPath:",self.labelpath)

        def __len__(self):
            """return number of points in our dataset"""
            return len(self.featurelist)

        def __getitem__(self, idx):
            """ Here we have to return the item requested by `idx`
                The PyTorch DataLoader class will use this method to make an iterable for
                our training or validation loop.
            """
            if sys.platform == "win32":
                imagepath =   self.featurepath+"\\"+ "Feature" +str(idx)+".npy"
                labelpath =   self.labelpath+"\\"+ "label"+str(idx)
            else:
                imagepath =   self.featurepath+"/"+ "Feature" +str(idx)+".npy"
                labelpath =   self.labelpath+"/"+ "label"+str(idx)
            img = np.load(imagepath, allow_pickle=True)
            img = img.astype('float32')
            

            #Below is to read and retrieve its contents, rb-read binary
            with open(labelpath, "rb") as f:
                label = pickle.load(f) 
                labels_2 = np.array(label[0]).astype('float32')
                labels_4 = np.array(label[1]).astype('float32')
                labels_8 = np.array(label[2]).astype('float32')
                labels_16 = np.array(label[3]).astype('float32')
            return img, labels_2 , labels_4 , labels_8, labels_16




    def define_dataset(self,train_test_switch, ProjectName):
        #global ProjectName
        dataset = None
        #print(self.ROOT)
        #try:
        if train_test_switch == "train":
            trainDatasetSaveplace = FileParentPath + "/Datasets/" + ProjectName + "/train"
            trainDataset = self.Dataset(trainDatasetSaveplace)
            #Now, you can instantiate the DataLoader:
            trainDataloader = DataLoader(trainDataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.n_cpu, drop_last=True)
            dataiter = iter(trainDataloader)
            trainDataset = dataiter.next()
            #testDataset  = transforms.ToTensor()
            trainDataset = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])                  
            dataset = trainDataset
            dataloader = trainDataloader
        
        elif train_test_switch == "test":
            #DEFINING TEST DATA LOADER FOR TESTINGs
            if sys.platform == "win32":
                testDatasetSaveplace = FileParentPath + "\\Datasets\\" + ProjectName + "\\test"
            else:
                testDatasetSaveplace = FileParentPath + "/Datasets/" + ProjectName + "/test"

            #testDatasetSaveplace = self.DataFolder+ "/test"
            testDataset = self.Dataset(testDatasetSaveplace)
            testDataLoader = DataLoader(testDataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.n_cpu, drop_last=True )
            #This will create batches of your data that you can access as:
            testiter=  iter(testDataLoader)
            testDataset = testiter.next()
            #testDataset  = transforms.ToTensor()
            testDataset = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])                  
            dataset = testDataset
            dataloader = testDataLoader


        '''
        except:
            PrintException()
            print("Did you rebuild the train test data? Please check")
            #input("")
            pass
        '''
        #####################################################################################################################
        #       DATA COMPLETE       ->     NOW MACHINE LEARNING PART
        #assert dataset is not None
        #else: input("Dataset is None")

        return dataset, dataloader



def choose_custom_or_mnist_dataset(opt):


    WhichDatasetIsChosen = input("Custom dataset or example dataset?  Type: (C/e) Example Dataset is just mnist and without BC/Lac Labels at the moment")

    if WhichDatasetIsChosen.lower() == "c" or WhichDatasetIsChosen == "":
        #   Create COUSTOM Pytorch DATASET with features and labels----------------------------------------------------------------------------
        # Source: [21] https://stackoverflow.com/questions/56774582/adding-custom-labels-to-pytorch-dataloader-dataset-does-not-work-for-custom-data
        from os import listdir
        from os.path import isfile, join

        #opt.DataFolder = FileParentPath + "/data/Images/"
        print("Chosen Datafolder is :", opt.DataFolder)

    elif WhichDatasetIsChosen.lower() == "e":
        # Configure data loader
        saveplace = FileParentPath + "/data/mnist"
        os.makedirs(saveplace, exist_ok=True)
        opt.DataFolder = saveplace


        opt.MNISTDataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                saveplace,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.Resize(self.opt.img_size), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.opt.batch_size,
            shuffle=True,
        )





    return opt


def guessFiletype(filepath):
    #https://pypi.org/project/filetype/
    kind = filetype.guess(filepath)
    if kind is None:
        #print('Cannot guess file type!')
        return

    #print('File extension: %s' % kind.extension)
    #print('File MIME type: %s' % kind.mime)

    return kind.extension, kind.mime


def choose_DataFormat(DataHandler):
    from os import listdir
    filelist = [f for f in listdir(DataHandler.DataFolder)]
    #print(filelist)
    headerlist = [["filename","Filetype", "mime_type"]]
    Data = np.array(headerlist)
    #Checks, which kind of data is in the Folder
    #Maybe: Periodical updating async function to avoid loading the whole dir just to choose Datatype
    for DataPoint in filelist:

        filename = str(DataPoint)
        if sys.platform == "win32":
            filepath = DataHandler.DataFolder + "\\" + filename
        else:
            filepath = DataHandler.DataFolder + "/" + filename
        #print("filepath: ",filepath)
        '''
        import os
        filename, file_extension = os.path.splitext(filename)
        #print(filename, file_extension)
        '''
        try:
            file_extension, mime_type = guessFiletype(filepath)
        except Exception as e:
            #print(e)
            #print("get extension from OS Mod")
            filename, file_extension = os.path.splitext(filepath)
            mime_type = "oth"
        
        Data = np.concatenate((Data,[[filename,file_extension,mime_type]]),axis=0)  #add action to orderArray

    
    Textfile =0  #2D Textfiles
    Tablefile = 0
    Audiofile = 0
    Pictures = 0    # 3D (Monochrome), 5D (RGB),...
    Videos = 0  # 3D(5D) Pic + 1D Time/Frames
    Unknown = 0
    Application = 0

    InitialEntry = [["filename","Filetype","mime_type"]]
    TextFileList = np.array(InitialEntry)
    TableFileList = np.array(InitialEntry)
    AudiofileList = np.array(InitialEntry)
    PictureFileList = np.array(InitialEntry)
    VideoFileList = np.array(InitialEntry)
    ApplicationFileList = np.array(InitialEntry)
    UnknownFileList = np.array(InitialEntry)




    for filename,Filetype,mime_type in Data:
        #print("filename,Filetype,mime_type: ",filename,Filetype,mime_type )

        if "image" in mime_type or 'png' in Filetype or 'jpg' in Filetype or 'bmp' in Filetype: # PICTURE
            Pictures +=1
            PictureFileList = np.concatenate((PictureFileList,[[filename,Filetype,mime_type]]),axis=0)  
            DataHandler.maxvalue = 256.0

        elif  "audio" in mime_type:  # AUDIO
            Audiofile += 1
            AudiofileList = np.concatenate((AudiofileList,[[filename,Filetype,mime_type]]),axis=0)  

        elif "video" in mime_type:  #VIDEO
            Videos +=1
            VideoFileList = np.concatenate((VideoFileList,[[filename,Filetype,mime_type]]),axis=0)  

        elif "application" in mime_type:
            Application +=1
            ApplicationFileList = np.concatenate((ApplicationFileList,[[filename,Filetype,mime_type]]),axis=0)  
        
        elif  "txt" in Filetype or "odt" in Filetype:  # If the datatypes consists out of understood txtfiles then its a textfile
            mime_type = "text/"+ Filetype[1:]
            Textfile +=1
            TextFileList = np.concatenate((TextFileList,[[filename,Filetype,mime_type]]),axis=0)  
        elif  "csv" in Filetype:  # If the datatypes consists out of understood txtfiles then its a textfile
            mime_type = "table/"+Filetype[1:]
            Tablefile +=1
            TableFileList = np.concatenate((TableFileList,[[filename,Filetype,mime_type]]),axis=0)

        else:
            Unknown +=1     # File not known to system but in Folder
            UnknownFileList = np.concatenate((UnknownFileList,[[filename,Filetype,mime_type]]),axis=0)  
    
    print("These types of files were discovered....")
    print("Textfiles",Textfile)
    print("Tablefiles", Tablefile)
    print("Audiofiles",Audiofile)
    print("Pictures",Pictures)
    print("Videos",Videos)
    print("Application data types",Application)
    print("Unknown data types",Unknown)
    '''
    mnistSaveplace = FileParentPath + "/data/mnist"
    if DataHandler.DataFolder == mnistSaveplace:
        print("MNIST Dataset detected, so chosen DataTypeChoice defauts to MNIST Dataset.object")
        DataTypeChoice = 6
    else:
        DataTypeChoice = int(input("Can't compare Apples with Oranges... \n Please choose one data type: \n 0:Text, 1:Audio, 2:Pictures, 3:Videos, 4:Application, 5:Other, 6:CSV"))
    '''
    DataTypeChoice = int(input("Can't compare Apples with Oranges... \n Please choose one data type: \n 0:Text, 1:Audio, 2:Pictures, 3:Videos, 4:Application, 5:Other, 6:CSV"))
    Datalist = [TextFileList, AudiofileList,PictureFileList , VideoFileList, ApplicationFileList, UnknownFileList,TableFileList]
    ChosenDataFileList = Datalist[DataTypeChoice]
    chosenDataFormat = DataHandler.DataFormatlist[DataTypeChoice]
    DataHandler.max_Value = DataHandler.max_Value_List[DataTypeChoice]
    DataHandler.ChosenDataFileList = ChosenDataFileList
    DataHandler.chosenDataFormat = chosenDataFormat

    return DataHandler


def merge_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = file1
    image2 = file2

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result


def ReadInByteModeSerializeAndPack(filepath):

    try:
        print("Read with base Python in READ BYTE MODE")
        f = open(filepath, "rb")
        #print(f)
        #print(f.readline())

    except Exception as e:
        print(e)
        try:
            print("Read with base Python in READ MODE with guessed encodings")
            f = open(filepath, "r",encoding= encoding)

            print("Read with base Python in READ MODE with varoious encoding formats")
            UsualEncodings = ["utf-8"  ," Windows-1252","Windows-1251",  "Windows-1250","ISO-8859-1","ISO-8859-2","Shift-JIS"]

            for encoding in UsualEncodings:
                try:
                    print("Try encoding ", encoding)
                    f = open(filepath, "r",encoding= encoding)
                except :
                    pass
        except:
            pass


    #CONVERT all lines in txt to String Var
    myline = f.readline()
    TextfileString = str(myline)
    while myline:
        #print(myline)
        myline = f.readline()
        TextfileString+= str(myline)
    f.close()
    #print("TextfileString",TextfileString)

    #try getting the DataHandler
    
    #Textfile.columns = ["a", "b", "c", "etc."]
    #Textfile_bytes = pickle.dumps(Textfile) #maybe hex coding? better?!
    #print("Textfile",Textfile)
    #print("Textfile_bytes",Textfile_bytes)
    
    print("Constructing Binary")
    binary=TextfileString.encode(encoding= "utf-16")
    
    #print(binary)
    hextext = binary.hex()


    HilbertCurveTransformation, Number_of_iterations, Number_of_Dimensions = MpMPacker.bytesting_to_HilbertCurve(hextext)
    npOutputfile = HilbertCurveTransformation

    return npOutputfile


def LoadArbitraryFileAs_npArray(DataHandler,filespecs):
    #DONE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #print(filespecs)
    #print(type(filespecs))
    #print("Chosen data format is", FileList[-1])
    #Filename, File-extention, mime_type = str(filespecs[0]), str(filespecs[1]), str(filespecs[2])
    filename = filespecs[0]
    fileextension = filespecs[1]
    #Fileextension = filespecs[1]
    if filename == "filename":
        #The first row is ignored cause it has just the headers
        return None, None
    else:
        print("Filename =", filename)
        print("fileextension =", fileextension)

    #input()

    if DataHandler.chosenDataFormat == "txt":

        filepath = filename + fileextension
        print("filepath",filepath)
        
        #Check encoding
        try:

            blob = open(filepath, 'rb').read()
            m = magic.Magic(mime_encoding=True)
            encoding = m.from_buffer(blob)
            #print("encoding: ",encoding)
        except  Exception as e:
            print(e)
            print("Error while getting encoding from file")
            encoding = "utf-8"
            #maybe make workaround for 

        npOutputFile =  ReadInByteModeSerializeAndPack(filepath)
        #Maybe muss in die codierung das packing vom 2D Text to 3D Numpy Array
        #Cause this is a transformation from 1d to 2d  the coding is i for number of iterations and number of dimensions=2 forcreating the hilbert curve
        codierung = ["i","D","V"]   # x and y for every pic in the world V for value


    elif DataHandler.chosenDataFormat  == "csv":
        print("CSV detected... try to open file")
        filepath = filename + fileextension
        print("Filepath is", filepath)
        df = pd.read_csv(filepath, sep=',',engine='python')
        print("Dataframe loaded")
        print(df.head())
        print("now try to convert file to numpy array")
        npOutputFile = df.to_numpy()

        codierung = ["i","x", "y"]
        pass



    elif DataHandler.chosenDataFormat  == "aud":
        #if data is Audio, then pack it like a 3D Picture with a  Fractal Teppich
        #3Blue1Brown
        filepath = DataHandler.DataFolder+ filename 
        #print("filepath: ",filepath)
        #test = fileextension[1:]
        #print(fileextension)
        #sound = AudioSegment.from_file(filepath, format=fileextension)
        #play(sound)
        
        #input()
        pass
        


    elif DataHandler.chosenDataFormat  == "pic":
        try:
            if sys.platform == "win32":
                filepath = DataHandler.opt.DataFolder+"\\" + filename  #+ fileextension   #Load Image with Pillow
            else:
                filepath = DataHandler.opt.DataFolder+"/" + filename  #+ fileextension   #Load Image with Pillow
        except:
            PrintException()
            print("Assuming filespecs[0](filename) is already filepath")
            filepath = filename
            print("Filepath to file is: ", filepath)
            Folderlist = []
            if sys.platform == "win32":
                Folderlist = filepath.split("\\")
            else:
                Folderlist = filepath.split("/")
            filename = Folderlist[-1]
        DataHandler.filename = filename
        DataHandler.filepath = filepath

        # Open the image
        image = Image.open(filepath)
        
        # If the picture is in RGB or other multi-channel mode 
        #just seperate the channels and concatenate them from left to right
        ChannleDimension = len(str(image.mode)) # grey -> 1 chan , rgb 3 channle
        
        if DataHandler.opt.verbosity == True:
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
            if DataHandler.opt.verbosity: print(channel)
            channelcodierung.append(channel)
        
        C1, C2, C3, C4, C5, C6 = None, None, None, None, None, None
        channellist = [C1,C2,C3,C4,C5,C6]
        croppedChannelList = channellist[0:ChannleDimension-1]
        croppedChannelList = image.split()        
        initEntry = None
        stackedchannels = np.array(initEntry)
        for idx, channel in enumerate(croppedChannelList):
            PicNumpy = np.array(channel)
            if DataHandler.opt.verbosity == True:
                print(PicNumpy)
                print(PicNumpy.shape)

            if idx == 0:
                stackedchannels = PicNumpy
            else:
                #instead of stacking the numpy arrays side to side. stack it into 3d dimension 
                stackedchannels = np.dstack((stackedchannels,PicNumpy))
                #stackedchannels = np.concatenate((stackedchannels,PicNumpy),axis=1)
            
        
        if DataHandler.opt.verbosity: print(stackedchannels.shape)
        npOutputFile = stackedchannels
        codierung = ["x","y"]   # x and y for every pic in the world
        codierung.append(channelcodierung) # grey (x,y,g) # RGB (x,y,R,G,B)



    elif DataHandler.chosenDataFormat  == "vid":
        '''
        #https://stackoverflow.com/questions/45607882/how-to-read-video-file-in-python-with-audio-stream-on-linux
        #BLUEPRINT to get to the the corresponding Video/Audio Frames
        from moviepy.editor import *

        video = VideoFileClip('your video filename')
        audio = video.audio
        duration = video.duration # == audio.duration, presented in seconds, float
        #note video.fps != audio.fps
        step = 0.1
        for t in range(int(duration / step)): # runs through audio/video frames obtaining them by timestamp with step 100 msec
            t = t * step
            if t > audio.duration or t > video.duration: break
            audio_frame = audio.get_frame(t) #numpy array representing mono/stereo values
            video_frame = video.get_frame(t) #numpy array representing RGB/gray frame
        '''

        #Unpack it with ffmpeg
        # Pack video alongside audio with the Fractal Teppich
        # But pack every 3D/5D Frame onto a 3D MpM 
    


    #THESE OBJECTS ARE FOR NOW READ IN BYTE MODE AND SERIALZED LIKE TXT DATA
    elif DataHandler.chosenDataFormat  == "app" or "oth":
        filepath = DataHandler.DataFolder+ filename 

        #If its a pdf, or zip, or something else
        #if Filetype is not known, than use Dataserialization like pickle/numpy to convert the file to a bytestream
        # the converted Bytestream is folded into a 3D Matrix like Audio or text
        #print("filespecs: ",filespecs)




        #print("Creating npOutputFile")
        npOutputFile =  ReadInByteModeSerializeAndPack(filepath)

        #print("Defining Coding")
        codierung = ["i","D","V"]       #V for Value


    return npOutputFile, codierung


def IterateOverDataFolder(DataHandler):
    opt = DataHandler.opt
    opt.WhichBoxCounting = DataHandler.opt.WhichBoxCounting
    if opt.WhichBoxCounting.lower() == "s": BoxcountingType = "spacial box counting"
    else: BoxcountingType = "traditional box counting"

    if opt.SaveOrNotToSave.lower() == "y": EnDis = "enabled"
    else: EnDis = "disabled" 

    print("Iterate over specified Folder with ",BoxcountingType, "and saving ",EnDis)
    #Checks what is in Folder and takes just valid Files
    #DataHandler = choose_DataFormat(DataHandler)
    #print("Chosen DataFormat is",DataHandler.chosenDataFormat)
    DataHandler.Dataset = []

    from tqdm import tqdm
    pbar = tqdm(total=len(DataHandler.ChosenDataFileList))

    if opt.SaveOrNotToSave.lower() == "y" or opt.SaveOrNotToSave.lower() == "y" : 
        print("Here you have to declare how and where to save the boxcounts")
        #saveplace
        saveplace = askdirectory()
        #saveplace = DataHandler.opt.FileParentPath + "/data/generated_imgs/"+DataHandler.opt.filename[:-3] +"png"
        print("saving images at: ", saveplace)
    codierung = ""
    #DensityMap= np.array([[0.0,0.0],])
    #DataHandler.lastVariance = 0.0

    for index, filespecs in enumerate(DataHandler.ChosenDataFileList):    # Filelist = [[Filename,File-extention, mime_type],[.,.,.]]
        pbar.update(1)
        if filespecs[0] == 'filename': continue     # Jump over Header

        try:
            print("Loading file:", filespecs[0])

            npOutputFile, codierung = LoadArbitraryFileAs_npArray(DataHandler,filespecs )
            print("File loaded...loaded shape is", npOutputFile.shape, "  and coding", codierung)

            print("For spacial Boxcounting just concatenate channels side by side.")

            for index, channel in enumerate(codierung[-1]):
                if index == 0:
                    output = npOutputFile[:,:,index]
                else:
                    output = np.concatenate((output, npOutputFile[:,:,index]),axis = 1)

            npOutputFile = output

            print("Concatenated npOutputFile shape is: ", npOutputFile.shape)


            #MULTICORE APROACH
            #print("Beginn Multithread Boxcount Lacunarity feature extraction")
            if opt.WhichBoxCounting.lower() == "s": 
                BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadBoxcount(npOutputFile, DataHandler.maxvalue)
            else:
                BoxcountingType = "traditional box counting"
                BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadChoosingBoxcount(npOutputFile,DataHandler)
            
            import DataExplorer
            Visualizer = DataExplorer.Visualize()
            
            if opt.Show_results.lower() == "y" or opt.SaveOrNotToSave.lower() == "y":

                if opt.WhichBoxCounting.lower() == "s":
                    #if spacial boxcounting, then display images

                    if opt.SaveAsNpOrPic.lower() == "n": 
                        if sys.platform == "win32":
                            imagesaveplace = saveplace+ "\\"+"Image_"+ str(index)
                            Boxcountsaveplace = saveplace+ "\\"+"BcLac_"+ str(index)
                        else:
                            imagesaveplace = saveplace+ "/"+"Image_"+ str(index)
                            Boxcountsaveplace = saveplace+ "/"+"BcLac_"+ str(index)

                        np.save(imagesaveplace, npOutputFile)
                        #saving label
                        #CANT save List with diffrent sized np arrays with np.save -> use pickle as workaround
                        pickle.dump(BoxCountR_SpacialLac_map_Dict,open(Boxcountsaveplace,"wb"))
                        Visualizer.characterize_Image_and_plot(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict,  "n" , opt.Show_results.lower() )


                    elif opt.SaveAsNpOrPic.lower() == "p":
                        DoYouWantToShow = "n"
                        Visualizer.characterize_Image_and_plot(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict,  opt.SaveOrNotToSave, DoYouWantToShow )

                elif opt.WhichBoxCounting.lower() == "b":
                    print("If original Boxcount is chosen, there are just the counts and lacs... no pics")
                    print("Printing Boxcounts/Lacs")
                    Visualizer.Print_characterized_image(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict, opt.SaveOrNotToSave)

            else: 
                EnDis = "disabled"             


        except Exception as e:
            print(e)
            PrintException()
            input("Error loading this file, continue to next one")
            continue
    pbar.close()

    print(len(DataHandler.Dataset))
    return DataHandler



def balancing( DataHandler,BoxCountR_SpacialLac_map_Dict,precision,index, DensityMap,lastVariance):
    print(  type(DataHandler), type(BoxCountR_SpacialLac_map_Dict), type(precision), type(index), type(DensityMap), type(DataHandler.lastVariance) )
    ##### CALC PLACING CONDITION FOR EACH DATAPOINT(PICTURE)
    #maxiteration = 3 # cause boxsize 2,4,8,16 = 0,1,2,3 iteration # determines the maximum boxsize
    #breaking it down to 2 values
    sumBCR, sumLAK = BoxCountR_SpacialLac_map_Dict[1][0], BoxCountR_SpacialLac_map_Dict[1][1]             #BoxCountR_SpacialLac_map_Dict[4] #[0], BoxCountR_SpacialLac_map_Dict[4][1] 
    if DataHandler.opt.verbosity: print("BCR,sumLAK  MAP before sum", sumBCR,sumLAK)

    sumBCR, sumLAK = np.sum(sumBCR), np.sum(sumLAK)
    if DataHandler.opt.verbosity: print("sumBCR,sumLAK after sum", sumBCR,sumLAK)
    
    #Calc, if the next data in the dataset will balance it more or not
    DataHandler.placingcondition, DensityMap, DataHandler.lastVariance = DataHandler.CalcPlacingcondition(DensityMap, sumBCR,sumLAK,precision,DataHandler.lastVariance, index)

    return DataHandler.placingcondition, DensityMap, DataHandler.lastVariance




def Chunking_Normalizing_Saving(DataHandler, npOutputFile, BoxCountR_SpacialLac_map_Dict, Num_train,Num_test):
    print("Begin Chunking_Normalizing_Saving")
    maxChunkCount = (npOutputFile.shape[1]/DataHandler.opt.img_size[1]) * (npOutputFile.shape[0]/DataHandler.opt.img_size[0])   #Chunks the picture is broken down into
    #print("maxChunkCount", maxChunkCount)
    Chunks = [None] * int(np.ceil(maxChunkCount))     # round up the non full boxes, cause they will be reshaped by padding with zeros       
    start = time.time()

    BoxBoundriesY = [0,DataHandler.opt.img_size[0]]
    BoxBoundriesX = [0,DataHandler.opt.img_size[1]]

    iteration = 0       # is Boxsize 32 -> should be faster then more little boxsize
    Boxsize=[2,4,8,16,32,64,128,256,512,1024]
    scalingFaktor = 1.0 / float(Boxsize[iteration])
    #If we take a sclice from the BCRmap/LAKmap, the boxboundries have to  be scaled for maintaining spacial dimensions across scaling with iteration and Boxsize
    Scaled_BoxBoundriesY = [0,int(DataHandler.opt.img_size[0]*scalingFaktor)]
    Scaled_BoxBoundriesX = [0,int(DataHandler.opt.img_size[1]*scalingFaktor)]
    
    
    try:
        print("Train/Test Ratio: ", str(Num_train),"/",str(Num_test)  ,"=",str(Num_train/Num_test))
    except:
        PrintException()
        pass

    min_max_scaler = preprocessing.MinMaxScaler()

    for i in range(len(Chunks)):
        if DataHandler.opt.verbosity == True:
            print("Current box",i,"of all",len(Chunks),"boxes")
            print("Boxboundries: X, Y :" ,BoxBoundriesX,BoxBoundriesY)
            print("CuttedBoxsizeList", CuttedBoxsizeList)
            print("Converting BCRmap,LAKmap into Chunked Form")

        Chunks[i] = npOutputFile[BoxBoundriesY[0]:BoxBoundriesY[1],BoxBoundriesX[0]:BoxBoundriesX[1]] 
        #print("Chunks.shape", Chunks[i].shape)
        #input("HALLLT")
        CHUNKED_BoxCountR_SpacialLac_map_Dict = {}
        CuttedBoxsizeList = Boxsize[:DataHandler.maxiteration+1]
        #print("Current box",i+1,"of all",len(Chunks),"boxes")
        #SCALE PICS FROM O...1
        Chunks[i] = min_max_scaler.fit_transform(Chunks[i]) 



        if BoxBoundriesX[1] > npOutputFile.shape[1]  or BoxBoundriesY[1] > npOutputFile.shape[0]:
            Chunkshape = Chunks[i].shape
            if DataHandler.opt.verbosity: 
                print("Chunkshape and shape are Diffrent... reshaping")
            #shape = 
            #print("Shape:", shape)
            Chunks[i] =  DataHandler.reshape_Data(Chunks[i], Chunkshape, DataHandler.opt.img_size)

        newshape = ( 1,int(DataHandler.opt.img_size[0]),int(DataHandler.opt.img_size[1]) )
        Chunks[i] = np.reshape(Chunks[i], newshape)




            
        for it , currentboxsize in enumerate(CuttedBoxsizeList):
            if DataHandler.opt.verbosity: print("Iteration", it, " and currentBoxsize", currentboxsize)
            
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
            
            #Check if chopped up BCR AND LAK are the same shape
            if Scaled_BoxBoundriesX[1] > chunked_BCRmap.shape[1]  or Scaled_BoxBoundriesY[1] > chunked_BCRmap.shape[0]:
                chunked_BCRmap_shape = chunked_BCRmap.shape
                #if DataHandler.opt.verbosity: 
                
                #print("BCR LAK-shape and wantedshape are Diffrent... reshaping")
                wantedshape = ( int(float(DataHandler.opt.img_size[0]) *scalingFaktor), int(float(DataHandler.opt.img_size[0]) *scalingFaktor)  )
                chunked_BCRmap =  DataHandler.reshape_Data(chunked_BCRmap, chunked_BCRmap_shape, wantedshape)

                chunked_LAKmap_shape = chunked_LAKmap.shape
                chunked_LAKmap =  DataHandler.reshape_Data(chunked_LAKmap, chunked_LAKmap_shape, wantedshape)

            #MINMAX SCALER SCALES TO 0...1
            chunked_BCRmap, chunked_LAKmap = min_max_scaler.fit_transform(chunked_BCRmap) , min_max_scaler.fit_transform(chunked_LAKmap)
            
                

            newshape = ( int(DataHandler.opt.img_size[0]*scalingFaktor),int(DataHandler.opt.img_size[1]*scalingFaktor) )
            chunked_BCRmap = np.reshape(chunked_BCRmap, newshape)
            chunked_LAKmap = np.reshape(chunked_LAKmap, newshape)





            #SCALING----------------------------
            #Scale the values of the Arrays in a gaussian distrubution with mean 0 and diviation 1?!?!?!
            #chunked_BCRmap, chunked_LAKmap = preprocessing.scale(chunked_BCRmap) , preprocessing.scale(chunked_LAKmap) 

            #Normalize the Values betweeen -1...1----------------------------
            #chunked_BCRmap, chunked_LAKmap = preprocessing.normalize(chunked_BCRmap, norm='l1')  , preprocessing.normalize(chunked_LAKmap, norm='l1')  
            #print("loader Normalizing disabled ")



            #ADD SCALED BCR LAK TO LABELDICT
            CHUNKED_BoxCountR_SpacialLac_map_Dict[it] = [chunked_BCRmap, chunked_LAKmap]


        if DataHandler.opt.verbosity == True:
            showNPArrayAsImage(Chunks[i], "Current Chunk", "gray")
            #print("chunked_BCRmap",chunked_BCRmap)
            #print("chunked_LAKmap",chunked_LAKmap)
            #showNPArrayAsImage(chunked_BCRmap, "chunked_BCRmap", "gray")
            #showNPArrayAsImage(chunked_LAKmap, "LAKmap", "gray") 
            print("sleeping 1 min")
            time.sleep(60)                   
            

    


        #PROCESSING DONE------------ NOW FORMING FEATURES AND LABLES AND SAVE
        feature =  Chunks[i]
        label =  [
            np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[0]) , 
            np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[1]) , 
            np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[2]) , 
            np.array(CHUNKED_BoxCountR_SpacialLac_map_Dict[3]) 
            ]
        
        if DataHandler.opt.verbosity == True:
            print("feature",feature)
            print("label",label)

        if DataHandler.placingcondition == True:
            #if DataHandler.opt.verbosity: print("Append Chunk to TRAIN dataset")
            #print("Append Chunk to TRAIN dataset")
            #saveplace
            trainsaveplace = FileParentPath+ "/Datasets/" + DataHandler.opt.ProjectName + "/train/"
            #saving image
            imagesaveplace = trainsaveplace+ "/features/"+"Feature"+ str(Num_train)
            np.save(imagesaveplace, feature)
            #saving label
            labelsaveplace = trainsaveplace+ "/labels/"+"label"+ str(Num_train)
            #CANT save List with diffrent sized np arrays with np.save -> use pickle as workaround
            print("LableSaveplace for checking NUM train test", labelsaveplace)
            
            pickle.dump(label,open(labelsaveplace,"wb"))
            Num_train +=1
                
        else:
            testsaveplace = FileParentPath+ "/Datasets/" + DataHandler.opt.ProjectName + "/test/"
            #print("Append Chunk to TEST  dataset")
            #saving image
            imagesaveplace = testsaveplace+ "/features/"+"Feature"+ str(Num_test)
            np.save(imagesaveplace, feature)
            #saving label
            labelsaveplace = testsaveplace+ "/labels/"+"label"+ str(Num_test)
            print("LableSaveplace for checking NUM train test", labelsaveplace)
            #np.save(labelsaveplace, label)
            #CANT save List with diffrent sized np arrays with np.save -> use pickle as workaround
            pickle.dump(label,open(labelsaveplace,"wb"))
            Num_test +=1
        




        #After this Chunk set the new Borders of the new chunk for next turn

        if BoxBoundriesX[1] < npOutputFile.shape[1]:
            #NOTES IF FAIL WILL HAPPEN
            #maxindexX = self.opt.img_size[1]
            #maxindexY = self.opt.img_size[0]

            BoxBoundriesX[0] =BoxBoundriesX[0] + DataHandler.opt.img_size[1]
            BoxBoundriesX[1] = BoxBoundriesX[1] + DataHandler.opt.img_size[1]
            if DataHandler.opt.verbosity == True: 
                print("Move box into x direction")
                print("BoxBoundriesX", BoxBoundriesX)
        else:
            if DataHandler.opt.verbosity == True:
                print(BoxBoundriesY,"BoxBoundriesY") 
                print("move box into starting position in x-direction")
                print("move box into ydirection")
            BoxBoundriesX[0]=0
            BoxBoundriesX[1]=DataHandler.opt.img_size[1]
            BoxBoundriesY[0]+=DataHandler.opt.img_size[0]
            BoxBoundriesY[1]+=DataHandler.opt.img_size[0]
            
    if DataHandler.opt.verbosity: input("Press any key for next File")


    end = time.time()     
    print(round(end,3) - round(start,3), "seconds passed for chunking and Make Train Data for 1 File")
    return  Num_train, Num_test



#TODO: WITH prccess_file_forDataset you can make a ProcessPool of 4 and just pass the Density Map and last variance every 4th time'
#      you CANNOT use multiprocessing.cpu_count because then the balancing will be off and not cross platform validation
# OR: Think about another way to harness multiprocessing power. In the end the calcing just has to be done one time





def process_file_for_dataset(DataHandler, Num_train, Num_test,npOutputFile ):

    start = time.time()

    for idx ,chan in enumerate(DataHandler.codierung):
        print("Channel", chan)
        #if picture is in rgb or something else than grayscale, then split picture into every channel and proceeed
        try:
            current_npOutputFile_channel = npOutputFile[:,:,idx]
        except:
            PrintException()
            print("Assuming just Grayscale image, so no channeldimension")
            current_npOutputFile_channel = npOutputFile

        if DataHandler.opt.img_size[1] > current_npOutputFile_channel.shape[1]  or DataHandler.opt.img_size[0] > current_npOutputFile_channel.shape[0]:
            print("LOADED ARRAY AND WANTED SHAPE ARE DIFFERENT................................................ reshaping")
            #print("Shape:", shape)
            current_npOutputFile_channel =  DataHandler.reshape_Data(current_npOutputFile_channel, current_npOutputFile_channel.shape , DataHandler.opt.img_size)


        #when already multi processes, multithread just takes too long to init
        #BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.SinglethreadBoxcount(current_npOutputFile_channel,DataHandler.maxvalue)

        BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadBoxcount(current_npOutputFile_channel,DataHandler.max_Value)
        
        #internal handling for NUm train and NUm test# because features and lable idx is num train num test


        maxChunkCount = (npOutputFile.shape[1]/DataHandler.opt.img_size[1]) * (npOutputFile.shape[0]/DataHandler.opt.img_size[0])   #Chunks the picture is broken down into
        #print("maxChunkCount", maxChunkCount)
        Chunks = [None] * int(np.ceil(maxChunkCount))     # round up the non full boxes, cause they will be reshaped by padding with zeros 

        Chunking_Normalizing_Saving(DataHandler, current_npOutputFile_channel,BoxCountR_SpacialLac_map_Dict, Num_train, Num_test )

        if DataHandler.placingcondition:
            for idxidx in range(len(Chunks)):
                Num_train += 1
        else:
            for idxidx in range(len(Chunks)):
                Num_test += 1



        #DataHandler.placingcondition, DensityMap, DataHandler.lastVariance = balancing(DataHandler, BoxCountR_SpacialLac_map_Dict,int(DataHandler.precision),index, DensityMap,DataHandler.lastVariance)

        #Num_train, Num_test, DataHandler.placingcondition, DensityMap, DataHandler.lastVariance = process_file_for_dataset(DataHandler,current_npOutputFile_channel, BoxCountR_SpacialLac_map_Dict, index)
        '''
        BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadBoxcount(current_npOutputFile_channel,DataHandler.max_Value)
        #print(BoxCountR_SpacialLac_map_Dict)
        #input("Halt")
        DataHandler.placingcondition, DensityMap, DataHandler.lastVariance = balancing(DataHandler, BoxCountR_SpacialLac_map_Dict,int(DataHandler.precision),index, DensityMap,DataHandler.lastVariance)
        Num_train, Num_test = Chunking_Normalizing_Saving(DataHandler, current_npOutputFile_channel,BoxCountR_SpacialLac_map_Dict, Num_train, Num_test, DataHandler.placingcondition  )
        '''
    

    end = time.time()
    print(round(end,3) - round(start,3), "seconds for Processing file for dataset")



def make_train_test_Datasets_multicore(DataHandler,opt):

    Num_test = 0
    Num_train = 0
    firsttime = True

    DensityMap= np.array([[0.0,0.0],])
    DataHandler.lastVariance = 0.0
    original_shape = DataHandler.opt.img_size

    #counter = 0
    #train_counter = 0
    #test_counter =0
    
    n = 4 # werden wohl bis max 16 begrenzen # ist 4 da beim slicen  richitg wäre  boxsize[:4] oder [:-7] 

    delete_dataset_from_last_time(FileParentPath, opt.ProjectName)

    if DataHandler.chosenDataFormat == "txt":
        import magic



    #Checks what is in Folder and takes just valid Files
    #DataHandler = choose_DataFormat(DataHandler)
    #print("Chosen DataFormat is",DataHandler.chosenDataFormat)
    DataHandler.Dataset = []

    from tqdm import tqdm
    pbar = tqdm(total=len(DataHandler.ChosenDataFileList))


    #DataHandler.precision = input("Set Precision for balancing train/test dataset: 0: No Balancing/50/50split     1...3...5:Coarse Balaning (many train/few test)      5...7...9: Fine balancing (few train/many test)   ")
    DataHandler.codierung = ""
    DensityMap= np.array([[0.0,0.0],])
    DataHandler.lastVariance = 0.0


    process1, process2, process3, process4, process5, process6, process7, process8, process9, process10, process11, process12, process13, process14, process15, process16, process17, process18, process19, process20, process21, process22, process23, process24 = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    processlist = [process1, process2, process3, process4, process5, process6, process7, process8, process9, process10, process11, process12, process13, process14, process15, process16]
    if opt.n_cpu == 0:
        processlist = [process1]
    else:
        processlist = processlist[:opt.n_cpu]

    #index = 0
    
    for index in range(1, len(DataHandler.ChosenDataFileList),opt.n_cpu):
        #for filespecs in DataHandler.ChosenDataFileList:    # Filelist = [[Filename,File-extention, mime_type],[.,.,.]]
        start = time.time()

        for ix, current_process in enumerate(processlist):
            pbar.update(1)
            filespecs = DataHandler.ChosenDataFileList[index]
            #dont need, cause we begin with startstep 1 
            #if filespecs[0] == 'filename': continue     # Jump over Header

            #sets placingcontition to true, when number is even
            #print("index is "+ str(index))
            #print("ix is"+ str(ix))
            num =  ix
            #print("Num for setting placingcondition is", num)
            DataHandler.placingcondition = (num % 2) == 0
            #print("placingcondition is"+ str(DataHandler.placingcondition))
            #time.sleep(5)
            #print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n \n \n \n \n \n ")



            print("Loading file:", filespecs[0])

            npOutputFile, DataHandler.codierung = LoadArbitraryFileAs_npArray(DataHandler,filespecs)

            print("File loaded...loaded shape is", npOutputFile.shape, "  and coding", DataHandler.codierung)
            print("DataHandler.codierung", DataHandler.codierung)
            print("np ouptut shape should be y,x,3 for rgb and y,x,1 for gray")
            print(npOutputFile.shape)

            processlist[ix] = multiprocessing.Process(target = process_file_for_dataset,args=(DataHandler, Num_train, Num_test,npOutputFile ))
            processlist[ix].start()
            '''
            if DataHandler.placingcondition:
                Num_train += 1
            else:
                Num_test += 1
            '''
            #cause process cant return easily. numtrain/test is calced externally and internally
            if DataHandler.placingcondition:
                #add the other channel
                try:
                    for i in range(len(DataHandler.codierung)):
                        Num_train += 1
                except:
                    print("Datahandler.codierung is"+ str(DataHandler.codierung))
                    print("maybe just only one coding, so no for loop able")
                    Num_train += 1

            else:   
                try:
                    for i in range(len(DataHandler.codierung)):
                        Num_test += 1
                except:
                    print("Datahandler.codierung is"+ str(DataHandler.codierung))
                    print("maybe just only one coding, so no for loop able")
                    Num_train += 1


            
            maxChunkCount = (npOutputFile.shape[1]/DataHandler.opt.img_size[1]) * (npOutputFile.shape[0]/DataHandler.opt.img_size[0])   #Chunks the picture is broken down into
            #print("maxChunkCount", maxChunkCount)
            Chunks = [None] * int(np.ceil(maxChunkCount))     # round up the non full boxes, cause they will be reshaped by padding with zeros 



            if DataHandler.placingcondition:
                try:
                    for idxidx in range(len(Chunks)-1):
                        Num_train += 1
                except:
                    PrintException()
                    print("Assuming there is just one Chunk")
                    #Num_train += 1

            else:
                try:
                    for idxidx in range(len(Chunks)-1):
                        Num_test += 1
                except:
                    PrintException()
                    print("Assuming there is just one Chunk")
                    #Num_test += 1



            index +=1

        #let the processes be joined
        for ix, current_process in enumerate(processlist):
            processlist[ix].join()
            print("Current_process is alive: {}".format(processlist[ix].is_alive()))
        end = time.time()
        print(round(end,3) - round(start,3), "seconds LOADING and PROCESSING "+ str(opt.n_cpu)+" files in Multicoremode")



    pbar.close()

    print(len(DataHandler.Dataset))

    return DataHandler









def make_train_test_Datasets(DataHandler):

    Num_test = 0
    Num_train = 0
    firsttime = True

    DensityMap= np.array([[0.0,0.0],])
    lastVariance = 0.0
    original_shape = DataHandler.opt.shape

    #counter = 0
    #train_counter = 0
    #test_counter =0
    
    n = 4 # werden wohl bis max 16 begrenzen # ist 4 da beim slicen  richitg wäre  boxsize[:4] oder [:-7] 

    delete_dataset_from_last_time(FileParentPath)

    if DataHandler.chosenDataFormat == "txt":
        import magic



    #Checks what is in Folder and takes just valid Files
    #DataHandler = choose_DataFormat(DataHandler)
    #print("Chosen DataFormat is",DataHandler.chosenDataFormat)
    DataHandler.Dataset = []

    from tqdm import tqdm
    pbar = tqdm(total=len(DataHandler.ChosenDataFileList))


    DataHandler.precision = input("Set Precision for balancing train/test dataset: 0: No Balancing/50/50split     1...3...5:Coarse Balaning (many train/few test)      5...7...9: Fine balancing (few train/many test)   ")
    codierung = ""
    DensityMap= np.array([[0.0,0.0],])
    lastVariance = 0.0

    for index, filespecs in enumerate(DataHandler.ChosenDataFileList):    # Filelist = [[Filename,File-extention, mime_type],[.,.,.]]
        pbar.update(1)
        if filespecs[0] == 'filename': continue     # Jump over Header

        try:
            print("Loading file:", filespecs[0])
            npOutputFile, codierung = LoadArbitraryFileAs_npArray(DataHandler,filespecs)

            print("File loaded...loaded shape is", npOutputFile.shape, "  and coding", codierung)
            print("codierung", codierung)
            print("np ouptut shape should be y,x,3 for rgb and y,x,1 for gray")
            print(npOutputFile.shape)
            

            for idx ,chan in enumerate(codierung):
                print("Channel", chan)
                #if picture is in rgb or something else than grayscale, then split picture into every channel and proceeed
                #
                current_npOutputFile_channel = npOutputFile[:,:,idx]


                if DataHandler.opt.shape[1] > current_npOutputFile_channel.shape[1]  or DataHandler.opt.shape[0] > current_npOutputFile_channel.shape[0]:
                    #if DataHandler.opt.verbosity: 
                    print("LOADED ARRAY AND WANTED SHAPE ARE DIFFERENT................................................ reshaping")
                    #shape = 
                    #print("Shape:", shape)
                    current_npOutputFile_channel =  DataHandler.reshape_Data(current_npOutputFile_channel, current_npOutputFile_channel.shape , DataHandler.opt.shape)


                BoxCountR_SpacialLac_map_Dict = BoxcountFeatureExtr.MultithreadBoxcount(current_npOutputFile_channel,DataHandler.max_Value)
                #print(BoxCountR_SpacialLac_map_Dict)
                #input("Halt")

                placingcondition, DensityMap, lastVariance = balancing(DataHandler, BoxCountR_SpacialLac_map_Dict,int(DataHandler.precision),index, DensityMap,lastVariance)
                Num_train, Num_test = Chunking_Normalizing_Saving(DataHandler, current_npOutputFile_channel,BoxCountR_SpacialLac_map_Dict, Num_train, Num_test, placingcondition  )

            
        except Exception as e:
            print(e)
            PrintException()
            input("Error loading this file, continue to next one")
            continue
    pbar.close()

    print(len(DataHandler.Dataset))

    return DataHandler

 
