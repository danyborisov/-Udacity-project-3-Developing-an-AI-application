import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import time 

import json
import torch
import torchvision
import argparse


from torchvision import models, datasets, transforms
from collections import OrderedDict
from PIL import Image

import functions
import net

# Collecting inputs for training
parser = argparse.ArgumentParser()
parser.add_argument('dir', type = str, help='Directory with images', default= '/home/workspace/ImageClassifier/flowers')
parser.add_argument('chck_dir', help='Direcrtory for the checkpoint save', default='/home/workspace/ImageClassifier/')
parser.add_argument('--arch', help='Pretrained model to be used vgg16 or vgg19: Default model vgg19', default='vgg19')
parser.add_argument('--gpu', help='Select GPU to train model', default = 'gpu')
parser.add_argument('--epochs', help='Number of epochs: Default 10 ', type=int, default=6)
parser.add_argument('--drop', help='Drop rate: Default 0.2 ', type=float, default=0.2)
parser.add_argument('--learning_rate', help='Learning rate: Default 0.0001', type=float, default=0.0001)
parser.add_argument('--hidden_layers',nargs='*', help='Number of hidden layers to add to classifier :Default 4096,1024,512', type=int, default=[4096,1024,512])
parser.add_argument('--output_size', help='Number of flowers classes to identify: Default 102', type=int, default=102)

args=parser.parse_args()

# Setting up the data for the model 
data_dir = args.dir

train_set, valid_set, test_set, trainloader, validloader, testloader = functions.image_preprocess(data_dir)
     
# Selecting parses chosen architecture and freezing hypermarameters 
arch = args.arch
if 'vgg' in arch:
    model = getattr(models, arch)(pretrained=True)
else:
    print('Choose from available architectures vgg16 of vgg19')

model.classifier = net.Model(args.output_size, args.hidden_layers, drop_rate = 0.2)
optimizer = torch.optim.Adam(model.classifier.parameters(),lr=args.learning_rate)
criterion = nn.NLLLoss()

print_every = 100
net.training_steps(model, trainloader, validloader, args.epochs, print_every, criterion, optimizer, args.gpu)

step_accuracy, step_loss = net.accuracy_check (model, validloader, criterion, args.gpu)
print(f"\nTest loss: {step_loss:.3f}")
print(f"Test accuracy: {step_accuracy:.3f}")

functions.save_checkpoint(model, train_set, optimizer, args.output_size, args.learning_rate, args.chck_dir, args.arch, args.epochs)