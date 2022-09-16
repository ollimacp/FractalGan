




#import pandas as pd

# Common Directories
import pathlib              #Import pathlib to create a link to the directory where the file is at.
FileParentPath = str(pathlib.Path(__file__).parent.absolute())
saveplace = FileParentPath + "/Datasets/"

#configfilepath = FileParentPath +"/config.pickle"

class config_file():
    def __init__(self):
        super(config_file, self).__init__()

        print("INIT------config_file----------------")
        #self.config_file = pd.Dataframe()
        self.ON_OFF_Switch = True
        self.ON_OFF_Switch_Hyperparams = False
        print("Altering opt is",self.ON_OFF_Switch)
        print("Altering Hyperparameters is",self.ON_OFF_Switch_Hyperparams)


    def set_Hyperparameterspace(self,ON_OFF_Switch_Hyperparams,HyperParameterspace):
        if ON_OFF_Switch_Hyperparams:
            HyperParameterspace = {
                'n_epochs':5,
                'lr':0.001, 
                'b1':0.9,
                'b2':0.999,
                'latent_dim':16,
            }    
        else:
            pass # and do nothing

        return HyperParameterspace

    def set_opt_parameters(self,ON_OFF_Switch,opt):

        if ON_OFF_Switch:
            print("Setting OPT PARAMETER")

            #opt.batch_size = 4
            #opt.lr = 0.001
            #opt.b1 = 0.9
            #opt.b2 = 0.999
            #opt.n_cpu = 
            #opt.sample_interval = 10000
            opt.verbosity = False

            #effective only in init_pop_phase
            #opt.Max_parallel_layers = 3
            #opt.Max_Lenght = 3
            #opt.max_happens = 5000

            #print(opt)

            #opt.No_latent_spaces = 1
            opt.UpdateMaskEvery = 50
            opt.breaker = False

            #opt.superres = False

            opt.noisebool = True
            #std = 0.001     #moderate disturbance       #cant see anything at all
            opt.std = 0.03
            #opt.std = 0.01      #Hard Disurbance
            #opt.std = 0.001     #moderate disturbance
            #opt.std = 0.0001    #light disturbance
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
            #opt.device = torch.device('gpu')
            opt.InpaintingParameters = {
                'opt': opt,
                'superresolution': (opt.superres, 2),
                'noise': (opt.noisebool, opt.std, opt.std_decay_rate),
                'mask': (opt.maskbool, opt.maskmean , opt.maskdimension),
                'Letterbox': (opt.LetterboxBool, opt.LetterboxHeight),
                'Pillarbox': (opt.PillarboxBool, opt.PillarboxWidth),

            }



        else:
            pass #and do nothing to opt

        return opt

    