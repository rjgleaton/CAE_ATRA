import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F


#Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
              
        return x


def main():
    #PATH TO EXPERIEMENT ONE
    PATH_1 = '2022.03.03 C2C12 ATRA differentiation'
    #PATH TO EXPERIMENT TWO
    PATH_2 = '2022.03.21 C2C12 ATRA IGF1 Differentiation'

    #Break images up into 64x64 tiles
    M = 64
    N = 64

    data = []

    #Images in path 1
    for file in os.listdir(PATH_1):
        f = os.path.join(PATH_1, file)
        if not f.endswith(".txt"):
            for filename in os.listdir(f):
                    image_path = os.path.join(f, filename)
                    if image_path.endswith(".tif"):
                        image = cv2.imread(image_path)
                        tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
                        data.append(tiles)
                        
    #Images in path 2
    for file in os.listdir(PATH_2):
        f = os.path.join(PATH_2, file)
        if not f.endswith(".txt"):
            for filename in os.listdir(f):
                    image_path = os.path.join(f, filename)
                    if image_path.endswith(".tif"):
                        image = cv2.imread(image_path)
                        tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
                        data.append(tiles)

    new_data = np.asarray(data)
    new_data = np.reshape(new_data, (110100, 64, 64, 3))
    new_data = np.transpose(new_data, (0,3,1,2))
    
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices = train_test_split(
        range(len(new_data)),
        test_size=0.15,
        random_state=14
    )

    # generate subset based on indices
    train_split = Subset(new_data, train_indices)
    test_split = Subset(new_data, test_indices)

    train_dataloader = DataLoader(train_split, batch_size=32, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(test_split, batch_size=32, num_workers=2, shuffle=True)

    #Instantiate the model
    model = ConvAutoencoder()
    print(model)

    #Loss function
    criterion = nn.BCELoss()

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = get_device()
    print(device)
    model.to(device)

    #Epochs
    n_epochs = 50

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0

        #Training
        for data in train_dataloader:
            images = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.float(), images.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
            
        train_loss = train_loss/len(train_dataloader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


def get_device():
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device



if __name__ == "__main__":
    main()
