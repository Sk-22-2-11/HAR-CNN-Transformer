# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 01:13:31 2023

@author: derek
"""
#%%
import numpy as np
import glob
import os
#import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import argparse

import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import threading
import webbrowser
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

#%%

class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(os.path.normpath(root_dir+'*/*.csv'))
        #self.folder = sorted(glob.glob(os.path.normpath(root_dir+"/*/")))  #for colab
        self.folder = glob.glob(os.path.normpath(root_dir+"/*/"))
        #print(self.folder)
        self.category = {self.folder[i].split('\\')[-1]:i for i in range(len(self.folder))}
        #print(self.category)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('\\')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        # normalize
        x = (x - 0.0025)/0.0119
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        x = torch.FloatTensor(x)
        #print(idx,sample_dir,y)
        return x,y
    
def load_data_n_model(dataset_name, model_name, root):
    classes = {'UT_HAR_data':7,'NTU-Fi-HumanID':14,'NTU-Fi_HAR':6,'Widar':22}
    if dataset_name == 'Widar':
        print('using dataset: Widar')
        num_classes = classes['Widar']

        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64, shuffle=True)        
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=256, shuffle=False)

        train_epoch = 12 #20
        if model_name == 'Lenet':
            print("using model: LeNet")
            model = Widar_LeNet(num_classes)
        elif model_name == 'Cnn':
            print("using model: CNN")
            model = CNN(num_classes)    
        elif model_name == 'CnnXY':
            print("using model: CNN_xy")
            model = CNN_xy(num_classes)
        elif model_name == 'SlowFast':
            print("using model: SlowFast")
            model = SlowFast(num_classes)

        
        return train_loader, test_loader, model, train_epoch
    
#%%

# Global lists to store metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def test(model, test_tensor, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    total = 0
    for inputs, labels in test_tensor:
        inputs, labels = inputs.to(device), labels.to(device).type(torch.LongTensor)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predict_y = torch.max(outputs, 1)
        test_acc += (predict_y == labels).sum().item()
        total += labels.size(0)

    test_acc = test_acc / total
    test_loss = test_loss / total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f"Validation accuracy: {test_acc:.4f}, loss: {test_loss:.5f}")

def train(model, train_tensor, test_tensor, num_epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        total = 0
        for inputs, labels in train_tensor:
            inputs, labels = inputs.to(device), labels.to(device).type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            _, predict_y = torch.max(outputs, 1)
            epoch_accuracy += (predict_y == labels).sum().item()
            total += labels.size(0)

        epoch_accuracy = epoch_accuracy / total
        epoch_loss = epoch_loss / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Epoch: {epoch+1}, Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.9f}")
        test(model, test_tensor, device)
'''
def test(model, test_tensor, device):
    model = model.to(device)
    
    test_acc = 0
    test_loss = 0
    for test_data in test_tensor:
        inputs,labels = test_data
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)            
                            
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)  

        loss = criterion(outputs,labels)                           
        test_loss += loss.item() * inputs.size(0)

        predict_y = torch.argmax(outputs,dim=1).to(device)
        test_acc += (predict_y == labels.to(device)).sum().item() / labels.size(0)
                              
    test_acc = test_acc/len(test_tensor)
    test_loss = test_loss/len(test_tensor.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))

    return

def train(model, train_tensor, test_tensor, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    print('train with epoch:{},test each epoch'.format(num_epochs))
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        #i=0
        #print('start training epoch {}'.format(epoch+1))
        for data in train_tensor:
            #print('start training img {}'.format(i+1))
            #i+=1
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)            
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            outputs = outputs.to(device)
            
            #print(outputs.shape, labels.shape)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()            
            epoch_loss += loss.item() * inputs.size(0)

            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        
        epoch_accuracy = epoch_accuracy/len(train_tensor)
        epoch_loss = epoch_loss/len(train_tensor.dataset)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))

        #with torch.no_grad():
        test(
        model= model,
        test_tensor=test_tensor,
        device= device
        )

    return
'''
#%%
class Basicblock(nn.Module):
    def __init__(self, input_dim, output_dim, stride):
        super(Basicblock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_block = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), stride=(stride, stride, stride), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(output_dim),        
        )

        self.conv_res = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=(1, 1, 1), stride=(stride, stride, stride), padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(output_dim),  
        )

    def forward(self, x):
        btn = self.conv_block(x)
        res = self.conv_res(x)
        out = self.relu(btn+res)
        return out


#%%
class Widar_LeNet(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (22,20,20)
            nn.Conv2d(22,32,6,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=1),
            nn.ReLU(True),
            nn.Conv2d(64,96,3,stride=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,96*4*4)
        out = self.fc(x)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True)
        )        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*4,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )
        

    def forward(self,x):
        #input = input.reshape(-1, 1, 22, 20, 20)
        x = x.permute(0,2,1,3)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.view(-1,256*5*4)
        out = self.fc(x)
        #print(out.shape)
        return out

    
class CNN_xy(nn.Module):
    def __init__(self, num_classes):
        super(CNN_xy,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True)
        )        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*4*2,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        #input = input.reshape(-1, 1, 22, 20, 20)
        y = x.permute(0,3,2,1)
        x = x.permute(0,2,1,3)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        
        x = x.view(-1,256*5*4)
        y = y.view(-1,256*5*4)
        out = torch.cat((x,y),dim=1)
        out = self.fc(out)
        #print(out.shape)
        return out
    

#%%

class Fast_rescnn(nn.Module):
    def __init__(self):
        super(Fast_rescnn, self).__init__()
        #input size: (batch,1,22,20,20)

        self.fast_en1 = Basicblock(1, 8, stride=1 )

        self.fast_en2 = Basicblock(8, 16, stride=1 )

        self.fast_en3 = Basicblock(16, 32, stride=2 )

        self.fast_en4 = Basicblock(32, 64, stride=1 )

        self.fast_en5 = Basicblock(64, 128, stride=2 )

        self.res_1 = nn.Conv3d(8, 8, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0) )
        self.res_2 = nn.Conv3d(16, 16, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0) )
        self.res_3 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0) )
        self.res_4 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0) )

    def forward(self, input):
        res = []

        x = self.fast_en1(input)
        #print(x.size())  
        res_1 = self.res_1(x)
        res.append(res_1)

        x = self.fast_en2(x)
        #print(x.size())  
        res_2 = self.res_2(x)
        res.append(res_2)

        x = self.fast_en3(x)
        #print(x.size())  
        res_3 = self.res_3(x)
        res.append(res_3)

        x = self.fast_en4(x)
        #print(x.size())  
        res_4 = self.res_4(x)
        res.append(res_4)

        x = self.fast_en5(x)

        #print('fast_size:{}'.format(x.size()))
        x = x.view(-1, 128*6*5*5)

        return x, res

class Slow_rescnn(nn.Module):
    def __init__(self):
        super(Slow_rescnn, self).__init__()
        #input size: (batch,1,11,20,20)

        self.slow_en1 = Basicblock(1, 16, stride=1 )

        self.slow_en2 = Basicblock(16+8, 32, stride=1 )
        
        self.slow_en3 = Basicblock(32+16, 64, stride=2 )
        
        self.slow_en4 = Basicblock(64+32, 128, stride=1 )
        
        self.slow_en5 = Basicblock(128+64, 256, stride=2 )
        
    def forward(self, input, res):

        x = self.slow_en1(input)
        #print(x.size(),res[0].size())
        x = torch.cat([x, res[0]],dim=1)

        x = self.slow_en2(x)
        #print(x.size(),res[1].size())
        x = torch.cat([x, res[1]],dim=1)

        x = self.slow_en3(x)
        #print(x.size())
        x = torch.cat([x, res[2]],dim=1)

        x = self.slow_en4(x)
        #print(x.size())
        x = torch.cat([x, res[3]],dim=1)

        x = self.slow_en5(x)
        #print(x.size())
        x = x.view(-1, 256*3*5*5)

        return x

class SlowFast(nn.Module):
    def __init__(self, num_classes):
        super(SlowFast, self).__init__()

        self.fast_conv1 = Fast_rescnn()
        self.slow_conv1 = Slow_rescnn()   
        
        self.fc = nn.Sequential(
                nn.Linear(128*6*5*5*2, 128, bias=False),
                nn.ReLU(),
                nn.Linear(128, num_classes, bias=False)
                )
    def forward(self, input):
        input = input.reshape(-1, 1, 22, 20, 20)
        #print(input.size())
        fast,res = self.fast_conv1(input)
        slow = self.slow_conv1(input[:, :, ::2, :, :],res)
        #print(slow.size(),fast.size())
        x = torch.cat([slow, fast], dim=1)
        x = self.fc(x)
        return x
    
#%%
root = './Data/' 
# model_name == 'Lenet'; 'Cnn';'CnnXY'; 'SlowFast'; 

train_loader, test_loader, model, train_epoch = load_data_n_model('Widar', 'Cnn', root)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#%%
from ptflops import get_model_complexity_info

macs, params = get_model_complexity_info(model, (22, 20, 20), as_strings=False,
                                         print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

#%%
train(
        model= model,
        train_tensor= train_loader,
        test_tensor= test_loader,
        num_epochs= train_epoch,
        learning_rate= 1e-3,
        criterion= criterion,
        device= device
    )
#%%

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    global train_accuracies, train_losses, test_accuracies, test_losses

    # Create the graph with subplots
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(train_losses))), y=train_losses,
                             mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(len(train_accuracies))), y=train_accuracies,
                             mode='lines+markers', name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(len(test_losses))), y=test_losses,
                             mode='lines+markers', name='Test Loss'))
    fig.add_trace(go.Scatter(x=list(range(len(test_accuracies))), y=test_accuracies,
                             mode='lines+markers', name='Test Accuracy'))

    return fig

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")
    
if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run_server(debug=True)