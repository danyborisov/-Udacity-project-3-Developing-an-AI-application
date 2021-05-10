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
from PIL import Image

import net
import functions

parser = argparse.ArgumentParser()
parser.add_argument('image', help='Image path for classification', default = '/home/workspace/ImageClassifier/flowers/test/12/image_04014.jpg')
parser.add_argument('chck_dir', help='Direcrtory for the checkpoint save', default = '/home/workspace/ImageClassifier/checkpoint1.pth')
parser.add_argument('--gpu', help='Select GPU for pediction. Default CPU for pre-trained model', default = 'cpu')
parser.add_argument('--top_k', type = int, help='Top classes to print', default = 1)
parser.add_argument('category_file', help='File with categories stored, should be JSON file', default = '/home/workspace/ImageClassifier/cat_to_name.json')

args=parser.parse_args()


model, optimizer = functions.load_chk(args.chck_dir, args.gpu)

if args.gpu == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'
    
model.to(device)
model.eval
print(f"Running on {device}\n")
      
#Loading and pre-rpocessing the image
class_image = functions.process_image(args.image)
class_image = torch.from_numpy(np.array([class_image])).float()
class_image.to(device)
class_out = model(class_image)

#Prediction
class_test = torch.exp(class_out)
top_p, top_class = class_test.topk(args.top_k, dim=1)

probabilities = top_p[0].tolist()
classes = top_class[0].tolist()

#Pulling indecies
l_idx = list(model.class_to_idx.items())
indicies = []

for i in range(len(classes)):
    for a in range(len(l_idx)):
        if classes[i] == l_idx[a][1]:
            indicies.append(l_idx[a][0])  
                
#Setting flower labeles
labels = []
for i in range(len(indicies)):
    labels.append(functions.label_name(args.category_file,indicies[i]))

print('\nTop Probabiliy in list: {}'.format(probabilities))
print('\nTop Classes in list: {}'.format(classes))
print('\nIndex in list: {}'.format(indicies))
print('\nFlower names in list: {}'.format(labels))
    