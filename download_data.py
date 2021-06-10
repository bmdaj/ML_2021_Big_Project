import paramiko
import numpy as np
from glob import glob
import os 
paramiko.util.log_to_file("paramiko.log")

# Open a transport
host,port = "io.erda.dk",22
transport = paramiko.Transport((host,port))

# Auth    
username,password = "LPhaxB7uEx","LPhaxB7uEx"
transport.connect(None,username,password)

# Go!    
sftp = paramiko.SFTPClient.from_transport(transport)

# Get list of files
files = sftp.listdir()


localPath = "/home/s200987/Documents/Jonas/appml/raw_data/"
localFiles = [os.path.basename(x) for x in glob(localPath+'*.txt')]
def Diff(li1, li2):
    return list(set(li1) - set(li2))

filesToGet = Diff(files, localFiles)


#%%
# Save the files to 
from tqdm import tqdm
for file in tqdm(filesToGet):
    data_string = sftp.open(file).read().decode('utf-8')
    data = np.array([np.array(i.split(' ')).astype('float') for i in data_string.split('\n')[:-1]]).transpose()
    np.savetxt("/home/s200987/Documents/Jonas/appml/raw_data/"+file,data)