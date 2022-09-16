
import numpy as np
import linecache
import sys
import os
import pathlib              #Import pathlib to create a link to the directory where the file is at.
import pickle
import random
import time
from collections import defaultdict
import statistics
from itertools import permutations
from tqdm import tqdm

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


class Network_Generator():
    #A evolutionary Network generator for FractalGAN

    def __init__(self,Parameter):
        super(Network_Generator, self).__init__()
        self.Sample_layerdict ={'0.125': [1,1,8], '0.25':[1,1,4], '0.5':[1,1,2], '1':[1,1,1], '2':[2,2,1], '4':[4,4,1], '8':[8,8,1]}
        self.layer_magnifications = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, ]
        if Parameter == None:
            self.img_size = 256
            self.latent_dim = 8
        else:    
            self.img_size = Parameter['img_size']
            self.latent_dim = Parameter['latent_dim']
        self.encoder_magnification = self.latent_dim / self.img_size
        self.decoder_magnification = self.img_size / self.latent_dim
        self.discriminator_magnification = 1.0/ self.latent_dim
        self.init_mag = 1.0
        self.parallel_multiplicator = [1,2,4,8]
        self.Channel_possibilitys = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32] #,32
        self.Max_Lenght = Parameter['Max_Lenght'] # discribes the max length of the network
        self.Max_parallel_layers = Parameter['Max_parallel_layers']
        self.opt = Parameter['opt']
        self.EnDeDis = None
        print("init finished")
    

    def init_all_magnification_trains(self, EnDeDis):
        if EnDeDis.lower() == "encoder":
            #constrain the allowed layer magnifications, so that the encoder doesn't see something like 0.5; 2; 0.5; 2...
            allowed_layermags = self.layer_magnifications[:-2] + self.layer_magnifications[:-2] + self.layer_magnifications[:-2] #only 0.125 to 2.0
            #allowed_layermags = self.layer_magnifications + self.layer_magnifications + self.layer_magnifications #only 0.125 to 1.0            
            print(f"encoder_magnification is {self.encoder_magnification}")
            print(f"allowed_layermags: {allowed_layermags}")

        elif EnDeDis.lower() == "decoder":
            #so that the encoder doesent see something like 0.5; 2; 0.5; 2; to infinity
            allowed_layermags = self.layer_magnifications[2:] +   self.layer_magnifications[2:] +self.layer_magnifications[2:]   #only from 1.0 to 8.0 for generation aspect
            #allowed_layermags = self.layer_magnifications + self.layer_magnifications + self.layer_magnifications #only 0.125 to 1.0            
            print(f"decoder_magnification: {self.decoder_magnification}")
            print(f"allowed_layermags: {allowed_layermags}")

        elif EnDeDis.lower() == "discriminator":
            #so that the encoder doesent see something like 0.5; 2; 0.5; 2; to infinity
            allowed_layermags = self.layer_magnifications[:4] + self.layer_magnifications[:4] + self.layer_magnifications[:4] #only 0.1258 to 1
            #allowed_layermags = self.layer_magnifications + self.layer_magnifications + self.layer_magnifications #only 0.125 to 1.0            
            #allowed_layermags = self.layer_magnifications[:-2] + self.layer_magnifications[:-2] + self.layer_magnifications[:-2] #only 0.125 to 2.0
            print(f"discriminator_magnification:  {self.discriminator_magnification}")
            print(f"allowed_layermags: {allowed_layermags}")

            
        Valid_combinations = []
        #Permutation list based on  https://www.geeksforgeeks.org/python-itertools-permutations/

        for current_length in range(1,self.Max_Lenght,1): 
            #permutations = list(permutations(allowed_layermags,self.Max_Lenght))
            permut = list(permutations(allowed_layermags,int(current_length)))
            for i, combo in enumerate(permut):
                aggregate_mag = 1.0
                for element in combo:
                    aggregate_mag = aggregate_mag * element
                
                if EnDeDis == "encoder" :
                    placing_condition =  aggregate_mag >= self.encoder_magnification
                elif EnDeDis == "discriminator":
                    placing_condition =  aggregate_mag >= self.discriminator_magnification
                elif EnDeDis == "decoder":
                    placing_condition = aggregate_mag == self.decoder_magnification
                
                if placing_condition == True:
                    Valid_combinations.append(combo)

        return Valid_combinations 


    def init_all_parallel_layers(self):
        Valid_combinations = []
        #Permutation list based on  https://www.geeksforgeeks.org/python-itertools-permutations/

        for possible_length in range(1,self.Max_parallel_layers+1):
            #print("possible lenght", possible_length)
            permutation_list = list(permutations(self.parallel_multiplicator,possible_length))
            
            for i, combo in enumerate(permutation_list):
                    Valid_combinations.append(combo)
            
        return Valid_combinations


    def init_all_Channel_train(self, net_lenght):
        Valid_combinations = []
        #Permutation list based on  https://www.geeksforgeeks.org/python-itertools-permutations/
        assert net_lenght <= len(self.Channel_possibilitys)
        permutation_list = list(permutations(self.Channel_possibilitys,net_lenght)) # C
        #print("Permutation list is", permutation_list)
        
        for i, combo in enumerate(permutation_list):
                Valid_combinations.append(combo)
        
        return Valid_combinations        


    def generate_random_Net(self,EnDeDis,No_latent_spaces,Valid_mag_train,Valid_parallel_layer_train ):

        def generate_random_layer(i, last_layer_index,EnDeDis, IN, OUT,No_latent_spaces ):
            layerlist = [  ]
            parallel_layer = []
            #print("lastlayerindex is",last_layer_index)
            #print("Layer:",i, "__IN:", IN, "__OUT:", OUT)

            if i == 0:
                #First layer, so first channel has to be 1 and output has to be more than 2 in case that the bcr/lac gets added
                if EnDeDis == "encoder":
                    IN = 1
                    if OUT <= 1:
                        OUT = 2

                elif EnDeDis == "decoder" or EnDeDis == "discriminator":
                    IN = No_latent_spaces

            elif i == last_layer_index and EnDeDis == "decoder":
                #Last Layer, so if decoder outputlayer has to be singular output and no parallel layers
                OUT = 1  #Just add singular ouput layer
                parallel_layer = [1] #and end in  1 channel output
            

            if parallel_layer == [1]: 
                #if parallel_layer is already chosen, then dont choose parallel layers
                pass
            else:
                #else
                parallel_layer =  list(Valid_parallel_layer_train[random.randrange(0,len(Valid_parallel_layer_train),1)])

            #layerelements=   gaussian Noise,         magnification           paralell layers   channels    Dropout/               dropout pct                 Batch norm
            layerlist = [ random.randint(0, 1 ), str(chosen_mag_train[i]), parallel_layer ,  [IN,OUT], [random.randint(0, 1 ), random.uniform(0.001, 0.3 ) ]   , random.randint(0, 1) ]
            #print(f"Generated random Layer is {layerlist}")

            return layerlist

        print(f"Number of all valid magnification trains are: {len(Valid_mag_train)}")
        chosen_mag_train = Valid_mag_train[random.randrange(0,len(Valid_mag_train),1)] #low high, step
        print(f"chosen_mag_train is {chosen_mag_train}")

        last_layer_index = len(chosen_mag_train) -1
        #print("Last layer index is", last_layer_index)
        Channel_train = self.init_all_Channel_train( last_layer_index+2)
        try:
            chosen_channel_train = Channel_train[random.randint(0,len(Channel_train))]
            print("chosen Channel train is", chosen_channel_train)
        except:
            PrintException()
            print("asume lenght is 0 so just one element")
            chosen_channel_train = list(Channel_train)
            print(f"chosen Channel train is: {chosen_channel_train}")

        LayerDescription = []
        for i, layer in enumerate(chosen_mag_train):
            # Output channels have to be the input channels of the next layer
            IN = chosen_channel_train[i]
            OUT = chosen_channel_train[i+1]
            layerlist = generate_random_layer(i,last_layer_index,EnDeDis, IN, OUT,No_latent_spaces)
            LayerDescription.append(layerlist)

        return LayerDescription


    def init_mating(self,opt):
        path = str(pathlib.Path(__file__).parent.absolute())        


        if sys.platform == "linux" or sys.platform == "darwin":
            path = path + "/models/GAN/" + opt.ProjectName +"/"        
        elif sys.platform == "win32":
            path = path + "\\models\\GAN\\" + opt.ProjectName +"\\"        


        print("Path is "+ path)
        self.Enc_netparams_file_list = []
        self.Dec_netparams_file_list = []
        self.Dis_netparams_file_list = []
        self.enc_dec_loss_list = []
        self.dis_loss_list = []

    
        #cause pickle.load(f) returns fail when netparams are loaded in cpu mode, when trained on gpu
        # taken from https://github.com/pytorch/pytorch/issues/16797
        import io
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else: return super().find_class(module, name)

        filecounter = 0
        n_encoder = 0
        n_decoder = 0 
        n_discriminator = 0
        #for FILE in os.listdir(path):
        #for FILE in os.scandir(path):
        '''
        #High memory usage with many files 
        #-> Workaround taken from https://stackoverflow.com/questions/58428835/how-to-reduce-memory-usage-of-python-os-listdir-for-large-number-of-files
        import glob
        folder_contents =  glob.iglob(path+"*.netparams")
        #for __, currentfile in zip(range(100),folder_contents):
        for currentfile in folder_contents:
        FILE = currentfile.replace(path,"")
        '''
        print("Exploring model files for evolutionary network architecture search:")
        for FILE in os.listdir(path):
            #print(f"File is named {FILE}")

            #for FILE in list(glob.glob)

            if FILE[-10:] == ".netparams":
                if filecounter % 50 ==0:
                    print(f" No {filecounter} of {len(os.listdir(path))}")

                try:
                    unixtime, lossvalue, Network_type = FILE.split("_")
                except:
                    lossvalue, Network_type = FILE.split("_")

                lossvalue  = lossvalue.replace("Loss","")
                lossvalue  = lossvalue.replace("---","")
                lossvalue = float(lossvalue)
                Network_type = Network_type.replace(".netparams","")

                with open(path+FILE, "rb") as f:
                    #print(f"Filesize of loaded netgenparameters f  {sys.getsizeof(f)}")
                    if self.opt.device == "cpu":
                        #print("load netparams into cpu")
                        NetParameters = CPU_Unpickler(f).load()
                    else:
                        #print("normal pickle load used")
                        NetParameters = pickle.load(f)

                    try:
                        NetParameters['LayerDescription'] = NetParameters['LayerDiscription']
                    except:
                        pass                    

                    #print(f"No of layers here {len(NetParameters['LayerDescription'])}")
                    if len(NetParameters['LayerDescription']) > opt.Max_Lenght:
                        print("Network too deep, continue with next model")
                        filecounter +=1
                        del NetParameters
                        continue

                    #print("NETPARAMETERS ARE " + NetParameters)


                if Network_type.lower() == "encoder":
                    n_encoder +=1
                    self.Enc_netparams_file_list.append([FILE, NetParameters, lossvalue])
                    self.enc_dec_loss_list.append(lossvalue)
                    #print(sys.getsizeof(self.Enc_netparams_file_list))
                
                elif Network_type.lower() == "decoder":
                    n_decoder += 1
                    self.Dec_netparams_file_list.append([FILE, NetParameters, lossvalue])
                    self.enc_dec_loss_list.append(lossvalue)

                elif Network_type.lower() == "discriminator":
                    n_discriminator +=1
                    Netparameters = pickle.load
                    self.Dis_netparams_file_list.append([FILE, NetParameters, lossvalue])
                    self.dis_loss_list.append(lossvalue)
                
                #whole_size = int(sys.getsizeof(self.Enc_netparams_file_list)) + int(sys.getsizeof(self.enc_dec_loss_list)) + int(sys.getsizeof(self.Dec_netparams_file_list)) + int(sys.getsizeof(self.Dis_netparams_file_list)) + int(sys.getsizeof(self.dis_loss_list)) 
                #print("Whole size is "+str(whole_size))

            filecounter +=1
                

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Number of Files: {filecounter},   encoder: {n_encoder},  decoder: {n_decoder}, discriminator: {n_discriminator},")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Calcing mean for Survivor selection
        #print(f"ENC_DEC mean lossLIST {self.enc_dec_loss_list}")

        self.enc_dec_mean_loss =  np.nanmean(self.enc_dec_loss_list, axis = 0)
        print(f"ENC_DEC mean loss is {self.enc_dec_mean_loss}")
        if opt.autoencoder == "off":
            self.dis_mean_loss = np.mean(self.dis_loss_list)

        print("Generated Parents Network architecture")
        print(f"Number of Encoder Models are {len(self.Enc_netparams_file_list)}")
        print(f"Number of Decoder Models are {len(self.Dec_netparams_file_list)}")
        assert len(self.Enc_netparams_file_list) > 0 and len(self.Dec_netparams_file_list) > 0 , "No Encoder/Decoder in Parent list"
        if opt.autoencoder == "off":
            print(f"Number of Discriminator Models are {len(self.Dis_netparams_file_list)}")    
            assert len(self.Dis_netparams_file_list) > 0 ,"No discriminator models found"

        return self



    def generate_children_from_parents(self, EnDeDis, Generation_limit,opt ):
        #NetParameters = {'LayerDescription': LayerDescription, 'input_shape': input_shape, 'SpacialBoxcounting':BoxcountEncoder, 'magnification':magnification, 'opt':opt, 'device': device}
        FileParentPath = str(pathlib.Path(__file__).parent.absolute())
        del_bad_models = input(f"Do you want to delete bad {EnDeDis} Models automaticly?  (y/N)")

        ################################################################
        #########     SURVIVOR SELECTION & ARCH COMPATIBILY CHECK
        ################################################################
        #If a network arch was performing worse/higher than the mean loss, than it'll not survive.
        #At the same time, the network lenght and the magnificatons are tracked, so that similar parents can be selected to create a valid child
        
        # Create dict with 2 keys, so that every Network is searchable by loss-value and latent dim size
        print(f"ENC_DEC mean loss is {self.enc_dec_mean_loss}")
        print(f"Generating {EnDeDis} models")

        if EnDeDis == "encoder":
            encoder_parents_dict = defaultdict(dict)
            enc_index = 0
            for element in self.Enc_netparams_file_list:
                filename, Netparameters, lossvalue = element
                #print("Filename: "+filename)
                encoder_opt = Netparameters['opt']
                print(f"loss for this model is {lossvalue}")
                if lossvalue < self.enc_dec_mean_loss:
                    print(" ENCODER network worthy of propagating")
                    encoder_parents_dict[str(encoder_opt.latent_dim)][str(lossvalue)] = Netparameters
                    enc_index += 1
                
                elif enc_index < Generation_limit:
                    print("Not worthy of propagating, but min pop has to be aquired")
                    encoder_parents_dict[str(encoder_opt.latent_dim)][str(lossvalue)] = Netparameters
                    enc_index += 1                    

                else:
                    print("Potential Parent is not performing enough")
                    if del_bad_models == "" or del_bad_models.lower() == "n":
                        pass
                    elif del_bad_models.lower() == "y":
                        print("Removing Encoder Model")
                        deletepath = f"{FileParentPath}/models/GAN/{encoder_opt.ProjectName}/{filename}"
                        os.remove(deletepath)
                        
                        modelpath =  f"{FileParentPath}/models/GAN/{encoder_opt.ProjectName}/{filename[:-10]}.model"
                        #os.remove(modelpath)
                        print(f"Deleted Netparams and model data for {filename[:-10]}")               
                        #time.sleep(5)
        elif EnDeDis == "decoder":
            decoder_parents_dict = defaultdict(dict)
            dec_index = 0
            for element in self.Dec_netparams_file_list:
                filename, Netparameters, lossvalue = element
                decoder_opt = Netparameters['opt']
                print(f"loss for this model is {lossvalue}")
                if lossvalue < self.enc_dec_mean_loss:
                    print(" DECODER network worthy of propagating")
                    print(f"latent dim is {decoder_opt.latent_dim}")
                    decoder_parents_dict[str(decoder_opt.latent_dim)][str(lossvalue)] = Netparameters
                    dec_index += 1

                elif dec_index < Generation_limit:
                    print("Not worthy of propagating, but min pop has to be aquired")
                    decoder_parents_dict[str(decoder_opt.latent_dim)][str(lossvalue)] = Netparameters
                    dec_index += 1        

                else:
                    print("Potential Parent is not performing enough")       
                    if del_bad_models == "" or del_bad_models.lower() == "n":
                        pass
                    elif del_bad_models.lower() == "y":
                        deletepath = f"{FileParentPath}/models/GAN/{decoder_opt.ProjectName}/{filename}"
                        os.remove(deletepath)
                        modelpath =  f"{FileParentPath}/models/GAN/{decoder_opt.ProjectName}/{filename[:-10]}.model"
                        #os.remove(modelpath)
                        print(f"Deleted Netparams and model data for {filename[:-10]}")
                

        elif EnDeDis == "discriminator":
            #Discriminator_parents_list = []
            discriminator_parents_dict = defaultdict(dict)
            dis_index = 0
            for element in self.Dec_netparams_file_list:
                filename, Netparameters, lossvalue = element
                dis_opt = Netparameters['opt']

                if lossvalue < self.dis_mean_loss:
                    print(" discriminator network worthy of propagating")
                    #Discriminator_parents_list.append([Netparameters,lossvalue])
                    print(f"latent dim is {dis_opt.latent_dim}")
                    discriminator_parents_dict[str(dis_opt.latent_dim)][str(lossvalue)] =  Netparameters
                    dis_index += 1

                elif dis_index < Generation_limit:
                    print("Not worthy of propagating, but min pop has to be aquired")
                    discriminator_parents_dict[str(dis_opt.latent_dim)][str(lossvalue)] =  Netparameters
                    dis_index += 1    


                else:
                    print("Potential Parent is not performing enough")       
                    if del_bad_models == "" or del_bad_models.lower() == "n":
                        pass
                    elif del_bad_models.lower() == "y":
                        try:
                                
                            deletepath = f"{FileParentPath}/models/GAN/{dis_opt.ProjectName}/{filename}"
                            os.remove(deletepath)
                            modelpath =  f"{FileParentPath}/models/GAN/{dis_opt.ProjectName}/{filename[:-10]}.model"
                            #os.remove(modelpath)
                            print(f"Deleted Netparams and model data for {filename[:-10]}")
                        except:
                            PrintException()
                            print("Could not delete this model")

        ################################################################
        #########     CROSSOVER & Sibling Mutation
        ################################################################
        '''  
        LayerDescription = [     gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch 
                                    [1, '1', [1]          , [1,4]  , [1, 0.112], 0 ],
                                    [1, '2', [1,2,4,8]    , [4,8]  , [1, 0.112], 1 ],
                                    [0, '2', [1,2,16]     , [8,16] , [1, 0.112], 1 ],
                                    [0, '4', [1,2,4,8,16] , [16,32], [1, 0.112], 1 ],
                                    [1, '1', [1,2,4,8,16] , [32,1] , [1, 0.112], 0 ],
                                ]         
        '''

        #Models are grouped by their latent dimension, to avoid layer mag fail
        chosen_latent_spaces_ori = ['2', '4', '8', '16', '32', '64', '128']
        # make sure, that latent spaces are always tinier than input, else there is no compression of data 
        chosen_latent_spaces = [x for x in chosen_latent_spaces_ori if int(x) < opt.img_size[0]]

        # The philosophy here is to get the best performing models and let them live on until they are too bad
        # and to mate them with the other survivors devided by the population_ratio
        population_ratio = 0.2 # the best 20% will survive/mate with the other

        if EnDeDis == "encoder":  
            self.Encoder_Netparameter_dict = defaultdict(dict)
            encoder_input= opt.channels
            encoder_output = opt.No_latent_spaces +1

        elif EnDeDis == "decoder":
            self.Decoder_Netparameter_dict = defaultdict(dict)
            decoder_input=  opt.No_latent_spaces +1
            decoder_output = opt.channels

        elif EnDeDis == "discriminator":
            self.Discriminator_Netparameter_dict = defaultdict(dict)
            discriminator_input = opt.No_latent_spaces
            discriminator_output = 1 #true or false


        for latent_dimension in chosen_latent_spaces:
            print(f"Processing latent dimension of {latent_dimension}")
            try:
                Best_Network_Arch_list = []
                Other_Network_Arch_list = []
        
                if EnDeDis == "encoder":      
                    subdict = encoder_parents_dict[latent_dimension]
                    self.encoder_magnification = int(latent_dimension) / self.img_size
                    magnification = self.encoder_magnification
                    
                elif EnDeDis == "decoder":
                    subdict = decoder_parents_dict[latent_dimension]
                    self.decoder_magnification = self.img_size / int(latent_dimension)
                    magnification = self.decoder_magnification

                elif EnDeDis == "discriminator":
                    subdict = discriminator_parents_dict[latent_dimension]
                    magnification = self.discriminator_magnification
                    self.discriminator_magnification = 1.0/ int(latent_dimension )
                
                sorted_loss_values = sorted(subdict)
                
                if len(sorted_loss_values) == 0:
                    print("Continue with next latent dimension, because no models found for latent dimension "+str(latent_dimension))
                    continue   

                print(f"sorted_loss_values: {sorted_loss_values}")
                print(f"Number models found {len(sorted_loss_values)}")
                population_ratio_index = int(float(len(sorted_loss_values))*population_ratio)
                print(f"population_ratio_index {population_ratio_index}")
                for index, lossvalue in enumerate(sorted_loss_values):
                    #spits out the network architecture with best loss(0.) to worst loss(>0.5)
                    #print("index, lossvalue, population_ratio_index "+ str(index) +" "+  str(lossvalue)+" "+  str(population_ratio_index))
                    Parent_Netparameters = subdict[lossvalue]
                    if index <= population_ratio_index:
                        Best_Network_Arch_list.append(Parent_Netparameters)
                        Other_Network_Arch_list.append(Parent_Netparameters)
                    else:
                        Other_Network_Arch_list.append(Parent_Netparameters)

                print(f" Lenght of best networks is {len(Best_Network_Arch_list)} and Lenght of all other nets  are {len(Other_Network_Arch_list)}")    
                
                for model_index, new_model in enumerate(range(Generation_limit)):
                    Parent1 = Best_Network_Arch_list[random.randint(0,len(Best_Network_Arch_list)-1)]
                    Parent2 = Other_Network_Arch_list[random.randint(0,len(Other_Network_Arch_list)-1)]
                    
                    #Because some old models were saved with a typo, this has to be adressed, by loading value with typo and fixing it
                    try:
                        Parent1['LayerDescription'] = Parent1['LayerDiscription']
                    except:
                        pass

                    try:
                        Parent2['LayerDescription'] = Parent2['LayerDiscription']
                    except:
                        pass
                        
                    #print(f"Parent1['LayerDescription'] {Parent1['LayerDescription']}")
                    #print(f"Parent2['LayerDescription'] {Parent2['LayerDescription']}")
                    Child = []
                    channel_input = 1
                    channel_output = 1
                    last_layer_index = len(Parent1['LayerDescription'])-1
                    AggregateMagnification = 1.0    #init magnification

                    for layer_index, layer in enumerate(Parent1['LayerDescription']):
                        #print("layer_index", layer_index)
                        #print("layer", layer)        

                        #if netgenparameters of parent1 are longer than those of the 2nd one, then just take the layerdescription from parent 1
                        #print("len(Parent2['LayerDescription'])", len(Parent2['LayerDescription']))
                        if layer_index < len(Parent2['LayerDescription']):
                            True_OR_False = random.randint(0,1) #random true false value
                        else:
                            True_OR_False = 1                        
                        #print("TrueOrFalse", True_OR_False)

                        if True_OR_False == 1:
                            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = Parent1['LayerDescription'][layer_index]
                            if layer_index == 0:
                                pass
                            else:
                                channel_input = channel_output  #because the channel input has to be the channel output of layer before
                                channellist = [channel_input,channellist[-1]]
                                channel_output = channellist[-1]

                            if self.opt.batch_size == 1:
                                #cause no batch norm possible, when just having batchsize of 1
                                batchnorm_switch = 0

                            Child.append([gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch])

                        else:
                            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = Parent2['LayerDescription'][layer_index]        #layer_from_Parent2
                            if layer_index == 0:
                                pass
                            else:
                                channel_input = channel_output
                                channellist = [channel_input,channellist[-1]]
                                channel_output = channellist[-1]
                            
                            if self.opt.batch_size == 1:
                                #cause no batchnorm, when just having batchsize of 1
                                batchnorm_switch = 0

                            Child.append([gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch])

                        layermagn = float(layermagn)  #layermag is right after gaussian noise
                        AggregateMagnification = AggregateMagnification * layermagn 
                        
                        if layer_index == last_layer_index:
                            #print("last layer reached, checking for correct channel output and magnification")
                            ######
                            ## INSANITY CHECK 
                            #  Check for each Network type for spec input and output
                            #  encoder in =1 and out = opt.no_latent_spaces
                            #  decoder in = opt.no_latent_spaces out = grey=1
                            #  discriminator in = opt.np_latent spaces out = 1 (t/f)
                            #
                            #   Check Magnifications 
                            #   assert magnification == in/out or something like this
                            #   else: just adjust last layer with right magnification or adjust a mag 1 layer according to the nessecary mag
                            #####
                            lastlayer = Child[-1]

                            gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = lastlayer
                            
                            if self.opt.batch_size == 1:
                                batchnorm_switch = 0 #cause no batchnorm, when just having batchsize of 1
                            #print("Child arch so far", Child)


                            if EnDeDis == "encoder":  
                                #checking channels
                                channellist = [channellist[0],encoder_output]
                                #print("checking magnification")
                                #print(f"AggregateMagnification {AggregateMagnification}" )
                                #print(f"self.encoder_magnification {self.encoder_magnification}" )
                                if AggregateMagnification == self.encoder_magnification:
                                    #print("Correct magnification... pass on")
                                    pass
                                else:
                                    #print("Not Correct mag, recalcing correct magnification")
                                    oldlayermagn = layermagn
                                    layermagn =  self.encoder_magnification / AggregateMagnification
                                    layermagn = str(float(layermagn) * float(oldlayermagn))
                                    #print(f"oldlayermagn {oldlayermagn}" )
                                    #print(f"newlayermagn {layermagn}" )

                            elif EnDeDis == "decoder":
                                #checking channels
                                channellist = [channellist[0],decoder_output]
                                #checking magnification

                                #print("AggregateMagnification", AggregateMagnification)
                                #print("self.decoder_magnification", self.decoder_magnification)

                                if AggregateMagnification == self.decoder_magnification:
                                    #print("Correct magnification... pass on")
                                    pass
                                else:
                                    #print("Not Correct mag, recalcing correct magnification")
                                    oldlayermagn = layermagn
                                    layermagn =   self.decoder_magnification / AggregateMagnification
                                    layermagn = str(float(layermagn) * float(oldlayermagn))
                                    #print("oldlayermagn", oldlayermagn)
                                    #print("newlayermagn", layermagn)
                            
                            elif EnDeDis == "discriminator":
                                #checking channels
                                channellist = [channellist[0],discriminator_output]
                                #checking magnification not neccesary, cause adaptive average pooling takes the last output and pools it to a singular value ranging from 0 to 1

                            Child[last_layer_index]= gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch

                    #print(f"Child Layer Discription{Child}")
                    
                    if EnDeDis == "encoder":  
                        self.Encoder_Netparameter_dict[latent_dimension][str(model_index)] = Child
                        #print(f"Appending Child to Encoder Netparameter dict with latent_dim={latent_dimension} and model index {model_index} :")
                        #print(Child)
                        #print(self.Encoder_Netparameter_dict)
                    elif EnDeDis == "decoder":
                        self.Decoder_Netparameter_dict[latent_dimension][str(model_index)] = Child
                        #print(f"Appending Child to Decoder Netparameter dict with latent_dim={latent_dimension} and model index {model_index} :")
                        #print(Child)                        
                        #print(self.Decoder_Netparameter_dict)
                    elif EnDeDis == "discriminator":
                        #print(self.Discriminator_Netparameter_dict)
                        self.Discriminator_Netparameter_dict[latent_dimension][str(model_index)] = Child
                    
                    last_model_index = model_index
                    #if Sum models are exceeding 80% of generation, break, so best old models can survive mutated
                    if model_index > int(0.8*Generation_limit):
                        print("Generation limit reached... Breaking")
                        break

                '''
                if EnDeDis == "encoder":  
                    print(f"MATING DONE, lenght of new ENCODER modellist with {latent_dimension} is  {str(len(self.Encoder_Netparameter_dict[latent_dimension]))}")
                    print(f"TRY TO READ ONE LAYERDISCRIPTION {self.Encoder_Netparameter_dict[latent_dimension]['0']}")
                    #print(f"Encoder Netparameter dict {self.Encoder_Netparameter_dict}")
                elif EnDeDis == "decoder":
                    print(f"MATING DONE, lenght of new DECODER modellist with {latent_dimension} is  {str(len(self.Decoder_Netparameter_dict[latent_dimension]))}")
                    #print(f"Decoder Netparameter dict {self.Decoder_Netparameter_dict}")

                elif EnDeDis == "discriminator":
                    print(f"MATING DONE, lenght of new DISCRIMINATOR modellist with {latent_dimension} is  {str(len(self.Discriminator_Netparameter_dict[latent_dimension]))}")
                    #print(f"DISCRIMINATOR Netparameter dict {self.Discriminator_Netparameter_dict}")
                '''

                #####################################
                #       best models survive, mutate and reinitialize
                #####################################
                print("Proceed with best models' survival, mutatation and reinitializion")

                parallel_layer_possibility = self.init_all_parallel_layers()
                #print("parallel layer possibilitys", parallel_layer_possibility)

                for best_model in Best_Network_Arch_list:
                    last_model_index +=1
                    mutated_model = []
                    opt= best_model['opt']
                    latent_dimension  = opt.latent_dim
                    for layer in best_model['LayerDescription']:
                        
                        gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch = layer
                        if self.opt.batch_size == 1:
                            batchnorm_switch = 0 #cause no batchnorm, when just having batchsize of 1

                        ##mutation occurs on gaussian noise, parrallel_layers, dropout/pct, batchnorm
                        gaussian_noise = random.randint(0,1)
                        parallel_layers = list(parallel_layer_possibility[random.randint(0,len(parallel_layer_possibility)-1)])
                        #print("chosen parrallel layers", parallel_layers)
                        
                        #high prob that best models have already good dropout value, so mutated around the old value
                        mean, standard_dev = dropout[0], 0.001
                        new_dropout = [random.randint(0,1), float(np.random.normal(mean,standard_dev))]
                        if new_dropout[1] <= 0.0  or  new_dropout[1] >= 1.0:
                            #if dropout is a non valid value then just take the old value
                            pass
                        else:
                            dropout = new_dropout
                        batchnorm = random.randint(0,1)
                        
                        mutated_model.append([gaussian_noise , layermagn, parallel_layers , channellist, dropout ,batchnorm_switch])
                    
                    #print("Last_model_index"+ str(last_model_index))
                    if EnDeDis == "encoder":  
                        self.Encoder_Netparameter_dict[latent_dimension][str(last_model_index)] = mutated_model
                        #print(f"MUTATE DONE, lenght of new modellist is  {str(len(self.Encoder_Netparameter_dict[latent_dimension]))}")
                    elif EnDeDis == "decoder":
                        self.Decoder_Netparameter_dict[latent_dimension][str(last_model_index)] = mutated_model
                        #print(f"MUTATE DONE, lenght of new modellist is  {str(len(self.Decoder_Netparameter_dict[latent_dimension]))}")
                    elif EnDeDis == "discriminator":
                        self.Discriminator_Netparameter_dict[latent_dimension][str(last_model_index)] = mutated_model
                        #print(f"MUTATE DONE, lenght of new modellist is {str(len(self.Discriminator_Netparameter_dict[latent_dimension]))}")

                if EnDeDis == "encoder":  
                    return_dict = self.Encoder_Netparameter_dict
                    
                elif EnDeDis == "decoder":
                    return_dict = self.Decoder_Netparameter_dict
                    
                elif EnDeDis == "discriminator":
                    return_dict = self.Discriminator_Netparameter_dict



            except:
                PrintException()
                input("fail in assembling new models")


            if EnDeDis == "encoder":  
                print(f"Mating and mutating DONE, lenght of new ENCODER modellist with {latent_dimension} is  {str(len(self.Encoder_Netparameter_dict[latent_dimension]))}")
                print(f"TRY TO READ ONE LAYERDISCRIPTION {self.Encoder_Netparameter_dict[str(latent_dimension)]['0']}")
                #print(f"Encoder Netparameter dict {self.Encoder_Netparameter_dict}")
            elif EnDeDis == "decoder":
                print(f"Mating and mutating DONE, lenght of new DECODER modellist with {latent_dimension} is  {str(len(self.Decoder_Netparameter_dict[latent_dimension]))}")
                #print(f"Decoder Netparameter dict {self.Decoder_Netparameter_dict}")

            elif EnDeDis == "discriminator":
                print(f" MODEL SURVIVAL AND MATING DONE, lenght of new DISCRIMINATOR modellist with {latent_dimension} is  {str(len(self.Discriminator_Netparameter_dict[latent_dimension]))}")




        return return_dict

    def get_child_arch(self, EnDeDis , latent_dimension ,Childlist_index, Netparameter_dict):
        #LayerDescription = None
        LayerDescription = Netparameter_dict[latent_dimension][Childlist_index]

        return LayerDescription
