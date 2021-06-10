import numpy as np
import glob
from tqdm import tqdm

num_bands = '5'
num_params = 4
num_k_points = 31

import os
directory = '/home/s200987/Documents/Jonas/appml/raw_data/'

# List of data files
files = glob.glob1(directory,"*.txt")
# files = [file for file in files if float(file.split('_')[0][1:]) <= 15]
#%%
# We count the number of files finishing with ".txt"
num_files =  len(files) 
print('Number of files in directory: ',num_files)

# We create the arrays to store all parameters and frequencies
params = np.zeros((int(num_files), int(num_params)))  
frequencies = np.zeros((int(num_files),int(int(num_bands)*num_k_points)))

index = 0

# We run through all files in directory
    
for file in tqdm(files):
    
    # We pick all ".txt" files
    
    if file.endswith(".txt"):
        # We get the value of the parameters by stripping the filename text
        
        x = np.array((file.split(".tx")[0]).split("_"))
        x = np.array([e[1:] for e in x]).astype(float)
        params [index, :] = x
        
        # We load the files to numpy and flatten it, so it can be stored in (num_files, k_points) format
        
        freq = np.loadtxt(directory+file).flatten()
        frequencies [index,:] = freq
        
        index += 1
        
#%%
import pandas as pd

columns_params = ["n", "e", "r [nm]", "t[º]"]
df_params = pd.DataFrame(params, columns = columns_params)

#%%
columns_k_points = [] #Here we will store all column labels

for i in range(np.shape(frequencies)[1]):
    
    #We get the band number
    
    band = i//int(num_k_points)
    
    #We obtain the k_number in each of the bands
    
    k_number = i - band * num_k_points
    
    #We append the label of the column
    
    columns_k_points.append('Band_'+str(band)+"_k_"+str(k_number))
    
#%%
df_frequencies = pd.DataFrame(frequencies, columns = columns_k_points)
df_params.to_csv("params_data.csv", index=False)
df_frequencies.to_csv("frequencies_data.csv", index=False)