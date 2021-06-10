import paramiko
import numpy as np

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

# Save the files to 
from tqdm import tqdm
for file in tqdm(files):
    data_string = sftp.open(file).read().decode('utf-8')
    data = np.array([np.array(i.split(' ')).astype('float') for i in data_string.split('\n')[:-1]]).transpose()
    np.savetxt("/home/s200987/Documents/Jonas/appml/raw_data/"+file,data)