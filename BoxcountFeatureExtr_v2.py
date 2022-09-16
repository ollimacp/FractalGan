print("Importing boxcounting libary")

import numpy as np
from numba import jit #Numba translates Python functions to optemized machine code at runtime and results in significant speedups
import time
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


verbosity = False   # False: no text displayed ;  True: potentially useful printouts are shown for info/bugfixing.

#Numba translates Python functions to optemized machine code at runtime and results in significant speedups
#from numba import jit
from PIL import Image  #Imagemanipulation via pillow image module
import matplotlib.pyplot as plt # Python plotting libary for visualizing graphs, data and images
import matplotlib.image as img 

#Helper-function to show any np.array as a picture with a chosen title and a colormapping 
def showNPArrayAsImage(np2ddArray, title, colormap):
    plt.figure()                    #Init figure
    plt.imshow(np2ddArray,          #Gererate a picture from np.array and add to figure
            interpolation='none',
            cmap = colormap)
    plt.title(title)                #Add title to figure
    plt.show(block=False)           #Show array as picture on screen, but dont block the programm to continue.



'''
class spacialBoxcounting():
    def __init__(self):
        super(spacialBoxcounting, self).__init__()
        print("INIT------ENCODER----------------")

'''

#If you want to compare the jit-compiler to the standard python interpreter just un/comment all @jit(... lines.
@jit(nopython= True) # Set "nopython" mode for best performance, equivalent to @njit
def Z_boxcount(GlidingBox, boxsize,MaxValue):
    continualIndexes = GlidingBox/boxsize    # tells in which boxindex the given value is in a continual way
    Boxindexes = np.floor(continualIndexes)  # Round all boxindexes down to ints to get the boxindex of each value in the gliding box 
    
    # If a value is in a given box, the boxcount increases by 1, but no further, when another  value is in the same box
    unique_Boxes = np.unique(Boxindexes)    # numpy helper function to create a list of the unique values by discarding doubles 
    counted_Boxes = len(unique_Boxes)       # the lenght of the list of all unique indexes are all unique counted boxes within the gliding box and the z-range

    if verbosity == True:
        print(continualIndexes)
        print("Boxindex",Boxindexes)
        print("counted_Boxes",counted_Boxes)
    
    #CREATE  List of SumPixInBox for all boxes to calc lacunarity
    InitalEntry = [0.0]
    SumPixInBox = np.array(InitalEntry)

    #For tiny boxes it can be process Consuming
    #for every unique counted boxindex in the list of all unique boxindexes
    for unique_BoxIndex in unique_Boxes:
        #set element-wise to True, when a boxindex is equal to the chosen unique boxindex
        ElementsCountedTRUTHTABLE = Boxindexes == unique_BoxIndex
        #the sum of the True elements represent the count of datapoints/pixel/voxel within the chosen box
        ElementsCounted = np.sum(ElementsCountedTRUTHTABLE)
        #Append the list of ElementsCounted to the list of all Elementcounted-lists to calc lacunarity later
        SumPixInBox = np.append(SumPixInBox, ElementsCounted)
        
        if verbosity == True:
            print("unique_BoxIndex",unique_BoxIndex)
            print("ElementsCountedTRUTHTABLE",ElementsCountedTRUTHTABLE)
            print("ElementsCounted",ElementsCounted)

            
    # Because the lacunarity is calculated with the standard deviation of all elementscounted
    # and the not counted boxes have a value of 0 boxes counted in this box
    # we have to calc the number of empty boxes by subtracting all counted boxes from the total amount of possible boxes
    Max_Num_Boxes = int(MaxValue/ boxsize)
    Num_empty_Boxes = Max_Num_Boxes- counted_Boxes
    
    if Num_empty_Boxes <1:      
        pass
        # if there is are no empty boxes, just pass
    else:
        EmptyBoxes = np.zeros(Num_empty_Boxes)
        SumPixInBox = np.append(SumPixInBox, EmptyBoxes)
    # calcs the mean of the list of all counted datapoints/pixel/voxel within chosen boxes and then...
    mean = np.mean(SumPixInBox)	 # ...calcs the standard deviation of the same
    standardDeviation = np.std(SumPixInBox)
    #The lacunarity or spacial heterogenity  = (standard deviation/mean)^2 
    Lacunarity=np.power(standardDeviation/mean,2)

    if verbosity == True:
        print("Max_Num_Boxes",Max_Num_Boxes)
        print("Num_empty_Boxes",Num_empty_Boxes)
        print("mean",mean)
        print("standardDeviation",standardDeviation)
        print("die Lacunarity ist", Lacunarity)

    return counted_Boxes, Lacunarity

@jit(nopython= True) #False,forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def spacialBoxcount(npOutputFile, iteration,MaxValue):
    '''
    This function takes in a 2D np.array the iteration which determins the boxsize
    and the maximum possible value to set up the value range. 8-Bit -> 256, hexadez ->16
    
    The function returns a 2 channel-2d array containing the spacial boxcount ratio and the
    spacial lacunarity scaled down in size by 1/Boxsize[iteration]
    '''
    Boxsize=[2,4,8,16,32,64,128,256,512,1024]  #All boxsizes
    boxsize = Boxsize[iteration]               #specified boxsize

    #Init counting box at x=0,y=0 und z=0
    BoxBoundriesX = np.array([0,Boxsize[iteration]])
    BoxBoundriesY = np.array([0,Boxsize[iteration]])

    Boxcount = 0
    YRange, XRange  = npOutputFile.shape
    
    #The maximum index of box with given boxsize in x and y direction
    maxIndexY = YRange / boxsize 
    maxIndexY = int(maxIndexY)+1

    maxIndexX = XRange / boxsize 
    maxIndexX = int(maxIndexX)+1

    
    
    if verbosity == True:
        print( "XRange, YRange", XRange, YRange)
        print("maxIndexX: ",maxIndexX,"maxIndexY: ",maxIndexY)

    #Initialize the BoxcountRatio_map and spacial_lacunarity_map with zeros in correct shape
    BoxCountR_map = np.zeros((maxIndexY,maxIndexX))
    spa_Lac_map = np.zeros((maxIndexY,maxIndexX))


    while BoxBoundriesY[1]<=YRange:

        while BoxBoundriesX[1]<=XRange:
            #Set up Boxindex with boxsize
            indexY = int(BoxBoundriesY[0]/boxsize)
            indexX = int(BoxBoundriesX[0]/boxsize)
            
            #Define Gliding box with Boundries ... for ex.  Boxsize 4  -> Boxboundries [0,4],[4,8] -> geht auf für n² wie in bildern etc und stitching is möglich
            GlidingBox = npOutputFile[BoxBoundriesY[0]:BoxBoundriesY[1],BoxBoundriesX[0]:BoxBoundriesX[1]] 

            counted_Boxes, Lacunarity = Z_boxcount(GlidingBox, boxsize, MaxValue)
            
            #Despite counting the Boxes like in the original algorithm, the counts are normalized from 0...1
            #0 means there was nothing counted inside  and 1 means every possible box is filled in
            Max_Num_Boxes = int(MaxValue/ boxsize)
            counted_Box_Ratio = counted_Boxes / Max_Num_Boxes 
            
            BoxCountR_map[indexY,indexX] = counted_Box_Ratio
            spa_Lac_map[indexY,indexX] = Lacunarity

            #move box into x direction, while boxboundriesx are <= xrange
            BoxBoundriesX[0]+=Boxsize[iteration]
            BoxBoundriesX[1]+=Boxsize[iteration]
            
            if verbosity == True:
                print("indexX: ",indexX)
                print("indexY: ",indexY)
                print("BoxBoundriesX: ",BoxBoundriesX)
                print("BoxBoundriesY: ",BoxBoundriesY)
                print("GlidingBox: ", GlidingBox)
                print("counted_Boxes, Lacunarity.: ",counted_Boxes, Lacunarity)

                
        #By exit inner while loop, box has reached end of x axis in array, so reset boxboundriesX to start
        BoxBoundriesX[0]=0
        BoxBoundriesX[1]=Boxsize[iteration]
        #and increase the counting box in y direction by a boxsize to scan the next line
        BoxBoundriesY[0]+=Boxsize[iteration]
        BoxBoundriesY[1]+=Boxsize[iteration]

    BoxCountR_SpacialLac_map = [BoxCountR_map, spa_Lac_map]
    
    if verbosity == True:
        print(BoxCountR_map)
        print(spa_Lac_map)
        print("Iteration ", iteration, "calculation done")

    return BoxCountR_SpacialLac_map

def SinglethreadBoxcount(npOutputFile,maxvalue):
    BoxCountR_SpacialLac_map_Dict = {}
    BoxsizeDict={"2":0 ,"4":1,"8":2,"16":3,"32":4,"64":5,"128":6,"256":7,"512":8,"1024":9}

    #Cut to lenght
    Height , width = npOutputFile.shape
    Height , width = int(Height) , int(width) 
    BaseITERMinVal = min(16,Height , width  )
    BaseIteration = BoxsizeDict[str(int(BaseITERMinVal))] #without 0 there are 1 more processes 
    maxiteration =  BaseIteration +1    # to calc Lacunarity there have to be more than just one box into the z direction

    #iterations = [None] * maxiteration   
    start = time.time()

    for iteration in range(maxiteration):
        BoxCountR_SpacialLac_map = spacialBoxcount(npOutputFile, iteration,maxvalue )     
        BoxCountR_SpacialLac_map_Dict[iteration]= BoxCountR_SpacialLac_map

    end = time.time()
    print(round(end - start,3),"seconds for spacial boxcounting with ",iteration+1, "iterations/scalings")

    return BoxCountR_SpacialLac_map_Dict


def MultithreadBoxcount(npOutputFile,maxvalue):
    
    '''
    To gain another speedup in the sequential generated output, multi threading is used
    to calculate the spacial Boxcountratios/lacunaritys for each boxsize in a own thread.
    
    '''
    #MULTICORE APROACH
    #print("Beginn Multithread Boxcount Lacunarity feature extraction")
    BoxsizeDict={"2":0 ,"4":1,"8":2,"16":3,"32":4,"64":5,"128":6,"256":7,"512":8,"1024":9}

    #Cut to lenght
    Height , width = npOutputFile.shape
    Height , width = int(Height) , int(width) 
    BaseITERMinVal = min(16,Height , width  )
    BaseIteration = BoxsizeDict[str(int(BaseITERMinVal))] #without 0 there are 1 more processes 
    maxiteration =  BaseIteration +1    # to calc Lacunarity there have to be more than just one box into the z direction
    
    #source: [17]  https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python

    def BoxcountBoxsizeWorker(npOutputFile, iteration):
        #maxvalue = 256  # cause zheight is 0...255: 8-bit grayscale picture
        #adjust for every specific input
        BoxCountR_SpacialLac_map = spacialBoxcount(npOutputFile, iteration,maxvalue )     
        
        return BoxCountR_SpacialLac_map

    from threading import Thread
    
    #Create thread-class with ability to return a value, which is not possible in threading
    class ThreadWithReturnValue(Thread):
        def __init__(self, group=None, target=None, name=None,
                    args=(), kwargs={}, Verbose=None):
            Thread.__init__(self, group, target, name, args, kwargs)
            self._return = None
        def run(self):
            #print(type(self._target))
            if self._target is not None:
                self._return = self._target(*self._args,
                                                    **self._kwargs)
        def join(self, *args):
            Thread.join(self, *args)
            return self._return


    #Init number of needed threads
    threads = [None] * maxiteration      
    if verbosity: print("Generate ",maxiteration,"threads")

    start = time.time()

    for i in range(len(threads)):
        threads[i] = ThreadWithReturnValue(target=BoxcountBoxsizeWorker, args=(npOutputFile, i))
        threads[i].start()
        if verbosity == True:
            print("thread ",i+1," has started")

    BoxCountR_SpacialLac_map_Dict = {"iteration": np.array(["BoxcountRatio","spacialLacunarity"]) }
    for i in range(len(threads)):
        BoxCountR_SpacialLac_map = np.array(threads[i].join())
        BoxCountR_SpacialLac_map_Dict[i]= BoxCountR_SpacialLac_map
        if verbosity == True :
            #print(BoxCountR_SpacialLac_map)
            print(BoxCountR_SpacialLac_map.shape)
            print(type(BoxCountR_SpacialLac_map))
            print("Thread ",i," JOINED")

    end = time.time()

    print(round(end - start,3),"seconds for spacial boxcounting with ",i+1, "iterations/scalings")
    
    if verbosity:
        input("Press any key to continue with next file. \n Attention: verbosity adds much size to the output of jupyter notebook. If the file > 120'ish MB, jupyter notebook crashes. So use just for debugging for beware. ")
    
    return BoxCountR_SpacialLac_map_Dict




def MultithreadChoosingBoxcount(npOutputFile,DataHandler):
    
    '''
    CHOOSE BETWEEN Traditional boxcounting and spacial boxcounting
    To gain another speedup in the sequential generated output, multi threading is used
    to calculate the spacial Boxcountratios/lacunaritys for each boxsize in a own thread.
    
    '''
    maxvalue = DataHandler.maxvalue
    #MULTICORE APROACH
    #print("Beginn Multithread Boxcount Lacunarity feature extraction")
    BoxsizeDict={"2":0 ,"4":1,"8":2,"16":3,"32":4,"64":5,"128":6,"256":7,"512":8,"1024":9}

    #Cut to lenght
    #print("npoutputfile.shape", npOutputFile.shape)
    Height , width = npOutputFile.shape
    Height , width = int(Height) , int(width) 
    BaseITERMinVal = min(16,Height , width  )
    BaseIteration = BoxsizeDict[str(int(BaseITERMinVal))] #without 0 there are 1 more processes 
    maxiteration =  BaseIteration +1    # to calc Lacunarity there have to be more than just one box into the z direction
    
    #source: [17]  https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python

    def BoxcountBoxsizeWorker(npOutputFile, iteration):
        #maxvalue = 256  # cause zheight is 0...255: 8-bit grayscale picture
        #adjust for every specific input
        #BoxCountR_SpacialLac_map = spacialBoxcount(npOutputFile, iteration,maxvalue ) 
        maxvalue = DataHandler.maxvalue
  
        if DataHandler.opt.WhichBoxCounting.lower() == "s" or DataHandler.opt.WhichBoxCounting.lower() == "" :
            BoxCountR_SpacialLac_map = spacialBoxcount(npOutputFile, iteration,maxvalue )   
        elif DataHandler.opt.WhichBoxCounting.lower() == "b":
            IterationDict={"0":2 ,"1":4,"2":8,"3":16,"4":32,"5":64,"6":128,"7":256,"8":512,"9":1024}
            boxsize = IterationDict[str(iteration)]
            Box_Count, lacunarity =  Z_boxcount(npOutputFile, boxsize,maxvalue)
            BoxCountR_SpacialLac_map = [Box_Count, lacunarity]
        
        return BoxCountR_SpacialLac_map

    from threading import Thread
    
    #Create thread-class with ability to return a value, which is not possible in threading
    class ThreadWithReturnValue(Thread):
        def __init__(self, group=None, target=None, name=None,
                    args=(), kwargs={}, Verbose=None):
            Thread.__init__(self, group, target, name, args, kwargs)
            self._return = None
        def run(self):
            #print(type(self._target))
            if self._target is not None:
                self._return = self._target(*self._args,
                                                    **self._kwargs)
        def join(self, *args):
            Thread.join(self, *args)
            return self._return


    #Init number of needed threads
    threads = [None] * maxiteration      
    if verbosity: print("Generate ",maxiteration,"threads")

    start = time.time()

    for i in range(len(threads)):
        threads[i] = ThreadWithReturnValue(target=BoxcountBoxsizeWorker, args=(npOutputFile, i))
        threads[i].start()
        if verbosity == True:
            print("thread ",i+1," has started")

    BoxCountR_SpacialLac_map_Dict = {"iteration": np.array(["BoxcountRatio","spacialLacunarity"]) }
    for i in range(len(threads)):
        BoxCountR_SpacialLac_map = np.array(threads[i].join())
        BoxCountR_SpacialLac_map_Dict[i]= BoxCountR_SpacialLac_map
        if verbosity == True :
            #print(BoxCountR_SpacialLac_map)
            print(BoxCountR_SpacialLac_map.shape)
            print(type(BoxCountR_SpacialLac_map))
            print("Thread ",i," JOINED")

    end = time.time()

    print(round(end - start,3),"seconds for spacial boxcounting with ",i+1, "iterations/scalings")
    

    return BoxCountR_SpacialLac_map_Dict


