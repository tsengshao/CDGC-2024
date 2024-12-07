import numpy as np
import pandas as pd
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return

def load_data(fname, caselist):
    from sklearn.preprocessing import normalize
    # θ, tke, enstrophy
    # dθ/dt
    for icase in range(len(caselist)):
      casen = caselist[icase]
      print(fname%(casen))
      data=np.load(fname%(casen))
      th  = data['th_tz'].T
      ens = data['enstrophy_tz'].T
      tke = np.zeros(ens.shape) 
      tke[:,1:] = data['tke_tz'].T
      dth = np.gradient(th, axis=0)

      if icase==0:
          dthall = dth[:-1,:]
          thall  = th[:-1, :]
          ensall = ens[:-1, :]
          tkeall  = tke[:-1, :]
      else:
          dthall = np.concatenate( (dthall, dth[:-1,:]), axis=0 )
          thall  = np.concatenate( (thall,  th[:-1, :]),  axis=0 )
          ensall = np.concatenate( (ensall, ens[:-1, :]), axis=0 )
          tkeall = np.concatenate( (tkeall, tke[:-1, :]), axis=0 )

    
    # normalized
    th_norm = normalize(thall,axis=1, norm='max')
    ens_norm = normalize(ensall,axis=1, norm='max')
    tke_norm = normalize(tkeall,axis=1, norm='max')
    th_norm  = thall / np.max(thall, axis=1, keepdims=True)
    print(th_norm.shape, ens_norm.shape, tke_norm.shape)
   
    # combine multiple input
    #inputs = np.stack((th_norm, ens_norm, tke_norm), axis=1)
    inputs = th_norm[:,np.newaxis,:]
    
    return inputs, dthall, thall

    

    ## #print(data.shape)
    ## #eddy=np.concatenate((data[0,:,:,0],data[1,:,:,0]),axis=0)
    ## tke=np.concatenate((data[0,:,:,1],data[1,:,:,1]),axis=0)
    ## enst=np.concatenate((data[0,:,:,2],data[1,:,:,2]),axis=0)
    ## th=np.concatenate((data[0,:,:,3],data[1,:,:,3]),axis=0)
    ## th0=th

    ## # Create target dataset
    ## dthdt=np.zeros([1442,50],dtype='float')
    ## for i in range(720):
    ##     dthdt[i+1,:]=th[i+1,:]-th[i,:]
    ##     dthdt[i+1+720,:]=th[i+1+720,:]-th[i+720,:]

    ## # Normalization: L2-normalization
    ## tke=normalize(tke,axis=1, norm='max')
    ## enst=normalize(enst,axis=1, norm='max')*10.
    ## th=normalize(th,axis=1, norm='max')

    ## # Create input dataset
    ## inputs=np.stack((tke,enst,th),axis=1)

    ## return inputs, dthdt*10., th0

def loss_fn(y_pred, y):
    mse = torch.nn.functional.mse_loss(y_pred, y)
    return mse

def scoring(y_pred,y):
    target=np.zeros(len(y),dtype="float")
    for i in range(len(y)-1):
        target[i+1,:]=y[i,:]+y_pred[i,:]/10.
    target[0,:]=y[0,:]+y_pred[0,:]/10.
    rmse=np.sqrt(mean_squared_error(y_pred,y))
    corr=np.corrcoef(y_pred,y)[0,1]
    print('Correlation: '+str(corr))
    print('RMSE: '+str(rmse))
    return corr,rmse

class CNN1D(nn.Module):
    def __init__(self, in_channels):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(32*44, 64)
        self.fc2 = nn.Linear(64, 50)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x shape should be [batch_size, in_channels, 50]
        x = self.relu(self.conv1(x))
        #print("CONV1D")
        #print(x.shape)
        x = self.pool1(x)
        #print("MaxPool1D")
        #print(x.shape)
        x = self.relu(self.conv2(x))
        #print("CONV1D")
        #print(x.shape)
        x = self.pool2(x)
        #print("MaxPool1D")
        #print(x.shape)
        x = self.flatten(x)
        #print("flatten")
        #print(x.shape)
        x = self.relu(self.fc1(x))
        #print("FC")
        #print(x.shape)
        x = self.fc2(x)
        #print("FC")
        #print(x.shape)
        return x

model = CNN1D(in_channels=1)
summary(model, input_size=(32,1,50), device="cpu")


if __name__=='__main__':
    caselist = ['pbl_op_8dth_6tr', 'pbl_op_11dth']
    inputs, dthdt, th0 = load_data(config.datPath+'/tz_series_%s_all.npz', caselist) # inputs: tke, enst, th
    x_data = np.copy(inputs) # (1442, 3, 50)
    y_data = np.copy(dthdt)  # (1442, 50)

    print(x_data.shape)
    print(y_data.shape)

    # Training NN model
    print("===== Training 1DCNN model =====")

    # Use GPU
    #import tensorflow as tf
    #device_name = tf.test.gpu_device_name()
    #if device_name != '/device:GPU:0':
    #   raise SystemError('GPU device not found')
    #print('Found GPU at: {}'.format(device_name))

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #  Convert to tensor
    x_tensor = torch.tensor(x_data)
    y_tensor = torch.tensor(y_data)
    full_dataset = TensorDataset(x_tensor, y_tensor)

    # Split the dataset into 80% training and 20% testing
    set_seed(0)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    th0_train, th0_test = random_split(th0, [train_size, test_size])

    # Get the indices of each subset
    train_indices = train_dataset.indices
    test_indices = test_dataset.indices

    # Hyperparameters
    learning_rate = 0.00005
    epochs = 6000
    batch_size = 32
    in_channels = 1

    #  Dataloader
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load Fully-connected NN model to gpu
    vcnn = CNN1D(in_channels=in_channels).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(vcnn.parameters(), lr=learning_rate)

    # Loss weight
    weight = 1+torch.linspace(1,0.1,y_data.shape[1])**2
    weight *= 10
    weight = weight[None,:].to(device, dtype=torch.float)

    # Training loop
    train_losses = []
    test_losses = []
    min_loss = 99999999999

    for epoch in range(epochs):
        vcnn.train()
        tot_loss  = 0
        for batch_x, batch_y in train_loader:
            # Move the subsets to GPU
            batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device, dtype=torch.float)
            # Forward pass
            predictions = vcnn(batch_x)
            loss = loss_fn(predictions*weight, batch_y*weight)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            tot_loss += loss
            optimizer.step()

        tot_loss /= len(train_loader.dataset)

        test_loss = 0
        # testing loss
        for test_x1, test_y1 in test_loader:
            batch_x1, batch_y1 = test_x1.to(device, dtype=torch.float), test_y1.to(device, dtype=torch.float)
            pred=vcnn(batch_x1)
            test_loss += loss_fn(pred, batch_y1)
        test_loss /= len(test_loader.dataset)


        #if (epoch + 1) % 10 == 0:
        train_losses.append(tot_loss.item())
        test_losses.append(test_loss.item())
        updated = False
        if min_loss > tot_loss:
            updated = True
            min_loss = tot_loss
            torch.save(vcnn, config.datPath+'VVM-1DCNN.pkl')
        print (
            '[{:>5d}/{:>5d}]'.format(epoch+1, epochs),
            'Loss:{:>.2e}, '.format(tot_loss.item()),
            'updated = {:>5s}, min loss={:>.2e}'.format(str(updated),min_loss)
        )

    # Plot training and testing loss
    plt.plot(train_losses, label='Training Loss')
    #plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('./fig/loss.png',dpi=200)
    plt.show(block=True)

    #vcnn=torch.load('VVM-1DCNN.pkl').to(device)
    vcnn.eval()
    output=[]
    for test_x1, test_y1 in test_loader:
        batch_x1, batch_y1 = test_x1.to(device, dtype=torch.float), test_y1.to(device, dtype=torch.float)
        pred=vcnn(batch_x1)
        output.append(pred.detach().cpu().numpy())
    output=np.array(output)
    print(output.shape)




