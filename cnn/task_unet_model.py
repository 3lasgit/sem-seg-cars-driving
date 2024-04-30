
import torch as tch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import argparse

# Importing the data 

parser = argparse.ArgumentParser()
parser.add_argument('--lr_rate', default=0.001, type=int)
parser.add_argument('--n_epoch', default=50, type=int)
parser.add_argument('--batch_size', default=1, type=int)
args = parser.parse_args()


# Add saving and loading models for big epochs' training

from google.cloud import storage


# Initialiser le client GCS

#project_number = os.environ["CLOUD_ML_PROJECT_ID"]
project_id = "semantic-segmentation-on-kitti"
client = storage.Client(project=project_id)


bucket_name = 'data_kitti_driv_seg'
bucket = client.get_bucket(bucket_name)

    
from io import BytesIO

# Récupérer l'objet depuis le bucket
object_path = 'data/training_tensor.pt'
blob = bucket.blob(object_path)
# Télécharger les données de l'objet en mémoire
data = BytesIO(blob.download_as_string())

training_tensor = tch.load(data)
training_tensor.shape


# Constructing the dataset objects

from torch.utils.data import Dataset
class ImageMaskDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.shape[1]  # Nombre d'exemples dans le tensor data

    def __getitem__(self, index):
        # Extraire l'image et le masque correspondant à l'index donné
        image = self.data[0, index]  # Première dimension pour les images
        mask = self.data[1, index]   # Deuxième dimension pour les masques
        
        return image, mask
    
# Splitting data into training/test datasets

training_data, test_data = ImageMaskDataset(training_tensor[:,:160]), ImageMaskDataset(training_tensor[:,160:])

# Création du DataLoader
data_loader = tch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True)


# Building the CNN (U-Net)

import torch.nn as nn

class DoubleConv(nn.Module): # Creating a class merging the double conv
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),          # X_out=X_in cf formula applied with these parameters' values
            nn.BatchNorm2d(out_channels),                                                # keeps size
            nn.ReLU(inplace=True),                                                       # keeps size 
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),         
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )                                                                                # Keeps the same image size of the input

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.dconv_down1 = DoubleConv(in_channels, 64)        # keeps image size 
        self.dconv_down2 = DoubleConv(64, 128)                # keeps image size 
        self.dconv_down3 = DoubleConv(128, 256)               # keeps image size 
        self.dconv_down4 = DoubleConv(256, 512)               # keeps image size 
        
        self.maxpool = nn.MaxPool2d(kernel_size = 2)          # X_out=int((X_in/2) + 1)   # Caution : default stride is equal to kernel-size here
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # X_out=int(X_in*2)       # Fasten the process as it hasn't to learn weights unlike the convtranspose (which is so )
        
        self.dconv_up3 = DoubleConv(256 + 512, 256)          # keeps image size 
        self.dconv_up2 = DoubleConv(128 + 256, 128)          # keeps image size
        self.dconv_up1 = DoubleConv(128 + 64, 64)            # keeps image size

        self.conv_last = nn.Conv2d(64, out_channels, 1)      # keeps image size

    def forward(self, x): 
        conv1 = self.dconv_down1(x)          
        x = self.maxpool(conv1)     

        conv2 = self.dconv_down2(x)          
        x = self.maxpool(conv2)     

        conv3 = self.dconv_down3(x)          
        x = self.maxpool(conv3)     

        x = self.dconv_down4(x)    
        x = self.upsample(x)        
        # print('La taille de x est ', x.shape, 'et la taille de conv3 est ', conv3.shape)
        x = tch.cat([x, conv3], dim=1) 

        x = self.dconv_up3(x)
        x = self.upsample(x)
        # print('La taille de x est ', x.shape, 'et la taille de conv2 est ', conv2.shape)
        x = tch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        #  print('La taille de x est ', x.shape, 'et la taille de conv1 est ', conv3.shape)
        x = tch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out
    

unet_model = UNet(in_channels = 3, out_channels = 3)


# Training the model

# Définir la fonction de perte (criterion) et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = tch.optim.Adam(unet_model.parameters(), lr=args.lr_rate)

unet_model.train()


for epoch in range(args.n_epoch) :
    running_loss = 0.0 
    
    if epoch % 5 == 0 :
        local_model_path = "unet_model.pt"
        object_path = 'model/' + local_model_path
        blob = bucket.blob(object_path)
        
        # Download the model locally
        blob.download_to_filename(local_model_path)

        # Charger les poids du modèle depuis le fichier
        state_dict = tch.load(local_model_path)

        # Mettre à jour les paramètres du modèle avec les poids chargés
        unet_model.load_state_dict(state_dict)
        unet_model.train()
        
    for image, mask in data_loader :
        # Remettre à zéro les gradients
        optimizer.zero_grad()

        pred = unet_model(image)

        # Calculate the loss
        loss = criterion(pred, mask)

        # Backpropagation and update of the weights
        loss.backward()
        optimizer.step()

        # Calculate the whole loss of the epoch
        running_loss += loss.item()
        
    if epoch % 5 == 0 :
        local_model_path = "unet_model.pt"
        tch.save(unet_model.state_dict(), local_model_path)

        object_path = 'model/' + local_model_path
        blob = bucket.blob(object_path)
        blob.upload_from_filename(local_model_path)

    # Afficher la perte moyenne de l'époque
    print(f"Epoch [{epoch+1}/{args.n_epoch}], Loss: {running_loss/len(data_loader)}")

    
# Model saving

local_model_path = "unet_model.pt"
tch.save(unet_model.state_dict(), local_model_path)

object_path = 'model/' + local_model_path
blob = bucket.blob(object_path)
blob.upload_from_filename(local_model_path)