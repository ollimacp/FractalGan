


#DataObject = MpMPacker.bytesting_to_HilbertCurve(DataObject,Textfile_bytes)

from numba import jit
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from ast import literal_eval

@jit(nopython= False, forceobj=True)#äFalse,forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def Initialize_Hilbertcurve(Number_of_iterations, Number_of_Dimensions,HilbertCurve_length):
    #https://pypi.org/project/hilbertcurve/
    #https://github.com/galtay/hilbertcurve
    p= Number_of_iterations # p = Number of iterations
    N=Number_of_Dimensions # N= Number of Dimensions: defaults to 2  for x,y
    
    hilbert_curve = HilbertCurve(p, N)
    HilbertCurve_length -=1 # cause 0 is 1 
    X_Dimension, Y_Dimension = retrieve_XY_coords_from( HilbertCurve_length,hilbert_curve )
    #print("The 2D Plane has a Dimension of ",str(X_Dimension), X_Dimension)

    return hilbert_curve, X_Dimension, Y_Dimension

@jit(nopython= False, forceobj=True) #äFalse,forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def retrieve_XY_coords_from(index, hilbert_curve):
    coords = hilbert_curve.coordinates_from_distance(index)
    #print(f'coords(h={index}) = {coords}')
    X, Y = coords[0], coords[1]
    return X,Y

@jit(nopython= False, forceobj=True) #äFalse,forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def retrieve_Hilbertlength_from(X,Y):
    coords = [X,Y]
    dist = hilbert_curve.distance_from_coordinates(coords)
    #print(f'distance(x={coords}) = {dist}')
    return dist


#@jit(nopython= False, forceobj=True) # Set "nopython" mode for best performance, equivalent to @njit
def bytesting_to_HilbertCurve(hextext):
    #Has to parse every bit/Byte and code it onto a Hilbert-Array
    
    # Get some sense how big the Hilbert curve has to be
    hextext_lenght = len(hextext) # to determine the lenth of the input and decide how big the Hilbertcure has to be
    #print("Textfilebytelenght is :", hextext_lenght)
    Number_of_iterations = 1  # initualize

    #Defining the needed HilbertCurveLenght for given hextext
    found = False
    while found == False:
        HilbertCurve_length = 4** Number_of_iterations

        if HilbertCurve_length <= hextext_lenght:
            Number_of_iterations+=1

        elif HilbertCurve_length > hextext_lenght:
            #if Hilbertcurve is bigger than the textfilelenght, the number of Iterations is right
            found = True
            break


    print("Number_of_iterations:", Number_of_iterations)
    #numdim stays at 2
    Number_of_Dimensions = 2
    hilbert_curve, X_Dimension, Y_Dimension =  Initialize_Hilbertcurve(Number_of_iterations,Number_of_Dimensions,HilbertCurve_length)
    
    Numpy_Size = (X_Dimension+1,X_Dimension+1)  # Cause last element in hilbert curve is always at [1,0] and the size should be 1,1, cause HilbertCurve is a square

    #print("Numpy_Size: ",Numpy_Size)

    HilbertCurveTransformation = np.zeros(Numpy_Size)

    #print("hextext: ",hextext)
    #Dezimaltext = literal_eval(hextext)
    #print("Dezimaltext: ",Dezimaltext)

    print("Try to stream every byte of the textstring to an xy coordinate of a hilbertcurve plane")
    for index, Hexzahl in enumerate(hextext):
        #print("Index: ",index, "   Hexzahl",Hexzahl)
        #Hexzahl = Hexzahl.capitalize()
        #print("Index: ",index, "   Hexzahl",Hexzahl)
        Hexzahl = str(Hexzahl)
        if index == 0:
            continue    #If its the inital Entry, then continue to first real entry
        else:
            pass

        try:
            X, Y = retrieve_XY_coords_from(index, hilbert_curve)
            #Dezimalwert = literal_eval(Hexzahl)
            Dezimalwert = int(Hexzahl, 16)

            #print("Der Wert", Dezimalwert,"sollte an Stelle",X,Y)
            #PACK STREAM INTO 3D ARRAY
            HilbertCurveTransformation[X,Y] = Dezimalwert
            
            #input()
        except :
            print("Error while transforming 1D-value -> 2D-value, Datareversibility lost")
            input()
    #print(HilbertCurveTransformation)
    #print("HilbertCurveTransformation.shape",HilbertCurveTransformation.shape )

    return HilbertCurveTransformation, Number_of_iterations, Number_of_Dimensions