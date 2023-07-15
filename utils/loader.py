import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self, X_path, y_path):
        
        #input_path is path to the side and front images numpy array
        self.X = np.load(X_path)
        self.X = self.X.reshape(len(self.X), 3, 512, 512)
        self.y = np.load(y_path)
        self.y = self.y.reshape(len(self.y), 3, 512, 512)
        
        # preprocessing step to convert grey image to mask
        #self.data = np.array(self.data,dtype='float')/255.0
        #self.data[self.data < 1] = 0
        #self.data = 1 - self.data

        #self.data = np.expand_dims(self.data, axis = 1)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        
        X = self.X[i]
        y = self.y[i]
        
        X = torch.from_numpy(X).double()
        y = torch.from_numpy(y).double()

        return X,y
