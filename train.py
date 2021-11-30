# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:59:38 2021

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
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
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

def replication (paths,root_len):# solving data imbalance
    label = pd.read_csv(os.path.join(args.datapath, 'train.csv'),index_col='id')
    orilen = len(paths)
    for i in range(orilen):
        if label.loc[paths[i][-(len(paths[i]))+root_len+1:-11]][0] == 1:
            paths.append(paths[i])
            #paths.append(paths[i])
        #else: paths.append(paths[i])
    return paths

def random_transformation(filenames):# augmentation by transformation
    for i in range(len(filenames)):
        p = np.random.uniform(0,1)
        flip = 1
        if p < 0.5:
            flip = 0
        degree = np.random.uniform(-30,30)
        filenames[i] = [filenames[i],flip,degree]
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
 
#seed
seed = 41

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(seed)
    
# input all datas except of unlabel ones
train_folder = os.path.join(args.datapath,'train')
train_paths = GetFileName(os.path.join(args.datapath,'train'),len(train_folder))
train_paths = replication(train_paths,len(train_folder))
train_paths = random_transformation(train_paths)
train_dataset = GetDataSet('train',train_paths,train = True)

#construct data loader
import torch.utils.data as data
batch_size = 16
ratio = 0.2
size_train = int(len(train_dataset) * (1 - ratio))
size_val = len(train_dataset)-size_train
train_set, valid_set = data.random_split(train_dataset, [size_train,size_val])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

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

#set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
#set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.parameters(),lr = 0.0001)

#training
epochs = 10
best_valid_loss = 10
best_epoch = 0

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.type(torch.ByteTensor).to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)
        
        # Print the progress of our training
        #counter += 1
        #print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.type(torch.ByteTensor).to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            #counter += 1
            #print(counter, "/", len(val_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    torch.save(model.state_dict(), os.path.join(args.datapath,'checkpoint_'+str(epoch)+'_.pth'))
    if (valid_loss<best_valid_loss):
        best_valid_loss = valid_loss
        best_epoch = epoch
    # Print out the information
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))