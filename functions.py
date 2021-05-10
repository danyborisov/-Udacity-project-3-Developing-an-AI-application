import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import time 

import json
import torch
import torchvision

from torchvision import models, datasets, transforms
from collections import OrderedDict
from PIL import Image


# Setting up image directories
def image_preprocess(data_dir):
    
    #Selecting source image folders
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Using various transforms for Training and normalizing for ImageNet
    data_transforms_training = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])

    # Using the same transforms for Testing AND Validation, normalizing for ImageNet
    data_transforms_testing = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])   

    # Loading the datasets with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform = data_transforms_training)
    valid_set = datasets.ImageFolder(test_dir, transform = data_transforms_testing)
    test_set = datasets.ImageFolder(valid_dir, transform = data_transforms_testing)

    # Using the image datasets and the transforms, defined the dataloaders
    trainloader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader (valid_set, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = True)

    return train_set, valid_set, test_set, trainloader, validloader, testloader

def save_checkpoint(model, train_set, optimizer, output_size, learning_rate, chck_dir, arch, epochs):
                     
    model.class_to_idx = train_set.class_to_idx

    checkpoint = {'input_size': 25088,
              'output_size': output_size,
              'arch': arch,
              'learning_rate': learning_rate,
              'batch_size': 64,
              'classifier' : model.classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
    if chck_dir == None:
        torch.save(checkpoint, 'checkpoint1.pth')
    else:
        torch.save(checkpoint, chck_dir+'checkpoint1.pth')
        
def load_chk(chk_dir, gpu):
    if gpu: #letting the user chose GPU or CPU within the funcion and loading it on a device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
        
    checkpoint = torch.load(chk_dir)
    learning_rate_1 = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=learning_rate_1)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

def process_image(image):
    with Image.open(image) as im:
        #resize with thumbnail
        size = 256, 256
        im.thumbnail(size)
        
        #box – The crop rectangle, as a (left, upper, right, lower)-tuple.
        side = 256
        half_crop = 0.5*(side-224)
        im = im.crop((half_crop,half_crop, (side-half_crop), (side-half_crop)))
        
        #tranalating PIL to np and normalising to 0-to-1 colorchanel floats
        np_im = np.array(im)/255 
        
        #Normalising for networks input 
        np_im = (np_im - ([0.485, 0.456, 0.406]))/([0.229, 0.224, 0.225])
        
        #Transpose for PyTorch and return for imshow
        return np_im.transpose(2,0,1)

    
    # Function to match the flower category to a name from json
def label_name (category_file, x):
    with open(category_file, 'r') as f:
            category_to_name = json.load(f)  
            return category_to_name[x]    