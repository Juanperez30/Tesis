import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from read_dataset import data_from_name
from model import *
from train import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import copy

#==============================================================================
# Training settingss
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default='1',  help='seed value')
parser.add_argument('--model', type=str, default='our', metavar='N', help='model')
parser.add_argument('--dp', type=float, default=0, metavar='N', help='dropout-value (default: 0)')
#
parser.add_argument('--lr', type=float, default=2e-1, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--lr_decay', type=float, default='0.2', help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[50, 100, 200, 300], help='decrease learning rate at these epochs')
#
parser.add_argument('--epochs', type=int, default=400, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--dataset', type=str, default='pendulum', metavar='N', help='dataset name')
#parser.add_argument('--dataset', type=str, default='./data/solar.txt', metavar='N', help='dataset name')
#
parser.add_argument('--dim', type=int, default=0, metavar='N', help='prediction dimension of y')
#
parser.add_argument('--dy', type=int, default=20, metavar='N', help='dimension of y')
#
parser.add_argument('--batch', type=int, default=10, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--length', type=int, default=4, help='size of Attention')
#
parser.add_argument('--wd', type=float, default=2e-5, metavar='N', help='weight_decay L2 regulization (default: 1e-5)')
#
parser.add_argument('--gradclip', type=float, default=2e-8, help='gradient clipping')
#
parser.add_argument('--noise', type=float, default=0.0, help='noise of data')
#
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def add_channels(X):
	return X.reshape(X.shape[0], 1, 1,X.shape[1])
#******************************************************************************
# load data Reshape data into 4D tensor Samples x Channels x Width x Hight
#******************************************************************************
#X = data_from_name(args.dataset,args.noise)
X = pd.read_csv(args.dataset, sep=',', header=None).values

t,dx =X.shape[0],X.shape[1]
###### scale #####
Xmax, Xmin = np.max(X), np.min(X)
X = ((X-Xmin)/(Xmax-Xmin)+0.01)*0.9


##### split into train and test set #####
Xtrain = X[0:t-2*args.dy+2]
Xtest = X[t-2*args.dy+2:t-args.dy+1]
Y = X[:,args.dim]
Ytrain = []
j = 0
for i in np.arange(args.dy,t-args.dy+1+1,1):
    Ytrain.append(Y[j:i])
    j = j + 1
Ytrain = np.array(Ytrain)

Ytest = []
for i in np.arange(t-args.dy+1+1,t+1,1):
    Ytest.append(Y[j:i])
    j = j + 1
Ytest = np.array(Ytest)
##### add channel #####
Xtrain,Xtest = add_channels(Xtrain),add_channels(Xtest)
Ytrain,Ytest = add_channels(Ytrain),add_channels(Ytest)
###### transfer to tensor #####
Xtrain,Xtest = torch.from_numpy(Xtrain).float().contiguous(),torch.from_numpy(Xtest).float().contiguous()
Ytrain,Ytest = torch.from_numpy(Ytrain).float().contiguous(),torch.from_numpy(Ytest).float().contiguous()
#******************************************************************************
# Create Dataloader objects
#******************************************************************************
train_data = torch.utils.data.TensorDataset(Xtrain, Ytrain)
train_loader = DataLoader(dataset = train_data, batch_size = args.batch)# shuffle = True
#==============================================================================
# Model summary
#==============================================================================

model = AAL(dx, args.dy,args.length,args.dp)

#print('**** Setup ****')
#print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
#print('************')
#print(model)

#==============================================================================
# Start training
#==============================================================================
model = train(model, train_loader,lr=args.lr, learning_rate_change=args.lr_decay, epoch_update=args.lr_update,
            weight_decay=args.wd, num_epochs = args.epochs, gradclip=args.gradclip)
#******************************************************************************
# Prediction
#******************************************************************************
model.eval()

Ytmp = Ytest[0:1,:,:,0:args.dy-1]
for i in np.arange(0,args.dy-1,1):
    Xinput = Xtest[i:i+1,:,:,:].float()

    Ypred = model(Xinput,Ytmp.float())
    Ytmp = torch.cat((Ytmp[:,:,:,1:args.dy-1],Ypred[:,:,:,args.dy-1:args.dy]),dim=3)


Ypred = Ytmp.detach().numpy().reshape(Ytmp.shape[3])
Ytest = np.array(Y[t-args.dy+1:t])
#******************************************************************************
# Results
#******************************************************************************
Y = (Y/0.9 - 0.01) * (Xmax - Xmin) + Xmin
Ypred = (Ypred/0.9 - 0.01) * (Xmax - Xmin) + Xmin
Ytest = (Ytest/0.9 - 0.01) * (Xmax - Xmin) + Xmin

Pearson = np.corrcoef(Ypred, Ytest)[0,1]
RMSE = np.sqrt(sum((Ypred-Ytest)**2)/len(Ypred))
StdY = Y[t-2*args.dy+2:t]
RMSE = RMSE / np.sqrt(sum((StdY-sum(StdY)/len(StdY))**2)/len(StdY))

print(Ypred)
#******************************************************************************
# Save data
#******************************************************************************
#address = '/' + args.dataset
#if not os.path.exists(address):
#	print(1)
#	os.makedirs(address)

#np.save(address +'/arg.npy',np.array((t,args.dy)))
#np.save(address +'/Y.npy', Y)
#np.save(address +'/'+args.model+'_Ypred.npy', Ypred)
#np.save(address +'/value.npy',np.array((Pearson,RMSE)))
#******************************************************************************
# draw pic
#******************************************************************************
#legend
#plt.title("Pearson :"+ str(round(Pearson,4)) +"  RMSE :" + str(round(RMSE,4)))
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.figure(figsize=(80, 30))

#plt.xlim(xmin=0,xmax=t)
#plt.ylim(ymin=min(Y.min(),Ypred.min())-0.3, ymax=max(Y.max(),Ypred.max())+0.3)
# draw line
#plt.plot(np.arange(1,t-args.dy+2,1), Y[0:t-args.dy+1], color='blue', linestyle='-',marker = "o")
#plt.plot(np.arange(t-args.dy+1,t+1,1),Y[t-args.dy:t], label='True',color='green', linestyle='-',marker = "o")
#plt.plot(np.arange(t-args.dy+1,t+1,1), np.concatenate([Y[t-args.dy:t-args.dy+1],Ypred]),label='Prediction',color='red', linestyle='-',marker = "o")


#plt.show()

