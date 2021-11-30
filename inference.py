# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:59:39 2021

@author: sammy
"""
# padding test data, radomly transforming training data
# input all datas except of unlabel ones
import os
import numpy as np
import glob
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
parser.add_argument("-output", "--outputfile")
args = parser.parse_args()

def GetFileName(root,root_len):
    filenames = glob.glob(os.path.join(root, '*.png'))
    #check if is nan
    label = pd.read_csv(os.path.join(args.datapath, 'train.csv'))
    label = label.dropna()
    label = label['id']
    label = label.array
    #delete nan hand
    count = len(filenames) - 1
    while (count >= 0):
        if (filenames[count][-(len(filenames[count]))+root_len+1:-11] not in label):
            filenames.remove(filenames[count])
        count = count - 1
    return filenames

class GetDataSet(Dataset):
    def __init__(self,file,file_paths,train=True):
        self.file_paths = file_paths #file names of images
        self.train = train
        self.resize = transforms.Resize((512,512))
        self.transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomEqualize(1),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean=0.485,std=0.229)
                            ])
        self.num_samples = len(self.file_paths)
        self.filenames = glob.glob(os.path.join(os.path.join(args.datapath,file), "*.png"))
        self.label = pd.read_csv(os.path.join(args.datapath, 'train.csv'),index_col='id')
        self.train_folder = os.path.join(args.datapath,file)
        self.flip = 0
        self.degree = 0
            
    def __getitem__(self,idx):
        if(self.train): 
            [file_path,self.flip,self.degree] = self.file_paths[idx]
        else: file_path = self.file_paths[idx]
        
        rimg = torchvision.io.read_image(file_path,torchvision.io.ImageReadMode(1))
        rimg = self.transforms(rimg)
        if (self.train): rimg = self.tran(rimg)
        rimg = self.padding(rimg)
        
        if (self.train):
            return rimg,self.label.loc[file_path[-len(file_path)+len(self.train_folder)+1:-11]][0]
        else:
            return rimg
        
    def __len__(self):
        return self.num_samples

    def padding(self,img):          
        h = img.size()[0]
        w = img.size()[1]
        if w != 512 or h != 512:
            if h > w:
                img = F.pad(img,((h-w)/2),((h-w)/2),0,0)
            elif w < h:
                img = F.pad(img,(0,0,(w-h)/2),((w-h)/2),0,0)
            img = self.resize(img)
        return img
    
    def tran(self,img):
        if (self.flip == 1): img = torch.fliplr(img)
        img = transforms.functional.rotate(img,angle=self.degree,expand=True)
        return img

# input all datas except of unlabel ones
test_paths = glob.glob(os.path.join(os.path.join(args.datapath,'test'), '*.png'))
test_dataset = GetDataSet('test',test_paths,train = False)

#construct data loader
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

#get pretrained model using torchvision.models as models library
from torch.autograd import Variable
model = models.regnet_x_800mf(pretrained=True)
#turn off training for their parameters
for param in model.parameters():
    param.requires_grad = True
    
#change the number of input channels to 4 
weight1 = model.stem[0].weight.clone()
new_first_layer  = nn.Conv2d(1, model.stem[0].out_channels, kernel_size=model.stem[0].kernel_size, stride=model.stem[0].stride, padding=(3, 3), bias=False).requires_grad_()
new_first_layer.weight[:,:1,:,:].data[...] =  Variable(weight1[:,:1,:,:], requires_grad=True)
model.stem[0] = new_first_layer

#check model weight
print(model.stem[0].weight.size())

#create new classifier for model using torch.nn as nn library
classifier_input = model.fc.in_features
num_labels = 2 #PUT IN THE NUMBER OF LABELS IN YOUR DATA
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
#replace default classifier with new classifier
model.fc = classifier

# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Move model to the device specified above
model.to(device)

#upload the checkpoint file
state_dict = torch.load(os.path.join(args.datapath,'checkpoint_'+str(2)+'_.pth'))
print(state_dict.keys())

#load the state dictionary to model
model.load_state_dict(state_dict)

print(model)

#testing
#evaluation mode
model.eval()
result = []  #storing the result of prediction (probability of being positive/abnormal)
count = 0
with torch.no_grad(): #don't calculate the gradient
    for inputs in test_loader:
        # Move to device
        inputs = inputs.to(device)
        # Forward pass
        output = model.forward(inputs)
        count += 1
        print(count,'/',len(test_loader))
        #store the result
        result.append(np.exp(float((output[0][1]).cpu())))

#output the prediction
#generate list of image id
test_folder = os.path.join(args.datapath,'test')
image_id = []
patient_id = []
f_result = []
for i in range (len(test_paths)):
    image_id.append(test_paths[i])
count = 0
test_paths = glob.glob(os.path.join(os.path.join(args.datapath,'test'), '*.png'))
test_paths.append('nothing')
while count < len(test_paths)-1:
    stop = count + 1
    while test_paths[stop][-len(test_paths[stop])+len(test_folder)+1:-11]\
    == test_paths[count][-len(test_paths[count])+len(test_folder)+1:-11]:
        stop = stop + 1
    patient_id.append(test_paths[count][-len(test_paths[count])+len(test_folder)+1:-11])
    temp = 0
    for i in range (count,stop):
        temp = temp + result[i]
    f_result.append(temp/(stop-count))
    count = stop

#build data frame for pandas library
df = pd.DataFrame({'id':patient_id,
                  'label':f_result})
#construct csv file for submission
df.to_csv(args.outputfile,index=False)