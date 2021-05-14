import numpy as np
from scipy.stats import boxcox
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time
from lime import lime_tabular as ltb

class baselineRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,
                 batch_size=1,num_layers=1,batch_first=True,dropout=0.0,
                h0=None):
        super(baselineRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size=input_size,hidden_size=hidden_size,
                           num_layers=num_layers,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,output_size)
        self.h0 = h0

    def forward(self, x):
        x, h_n  = self.rnn1(x,self.h0)

        # take all outputs
        out = self.lin(x[:, :, :])

        return out

class baselineLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,
                 batch_size=1,num_layers=1,batch_first=True,dropout=0.0,
                 h0=None,
                 c0=None):
        super(baselineLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                           num_layers=num_layers,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,output_size)
        self.h0 = h0
        self.c0 = c0

    def forward(self, x):
        x, (h_n, c_n)  = self.rnn(x,(self.h0,self.c0))

        # take all outputs
        out = self.lin(x[:, -1, :])

        return out

class baselineGRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,
                 batch_size=1,num_layers=1,batch_first=True,dropout=0.0,
                h0=None):
        super(baselineGRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,output_size)
        self.h0 = h0

    def forward(self, x):
        # print(self.h0.shape)
        x, h_n  = self.rnn(x,self.h0)

        # take last cell output
        out = self.lin(x[:, :, :])

        return out

CSV_FILE = '/home/matt/data/Rain_In_Australia/weatherAUS.csv'
LOSS_PATH = 'losses/LSTM.csv'
MODEL_PATH = 'models/LSTM.pt'
df_labels_list = []
df_data_list = []
df = pd.read_csv(CSV_FILE)
list_idx = -1
for index, row in df.iterrows():
    if index == 0 or df.loc[index-1, 'Location'] != row['Location']:
        df_labels_list.append(np.array(row['RainTomorrow']))
        df_data_list.append(row['MinTemp':'RainToday'].to_numpy())
        list_idx += 1
    else:
        df_labels_list[list_idx] = np.vstack((df_labels_list[list_idx], np.array(row['RainTomorrow'])))
        df_data_list[list_idx] = np.vstack((df_data_list[list_idx], row['MinTemp':'RainToday'].to_numpy()))
        
for i in range(len(df_data_list)):
    for j in range(20):
        df_data_list[i][:,j] += 1 + (-1*min(df_data_list[i][:,j]))
        df_data_list[i][:,j] = np.diff(df_data_list[i][:,j],n=2,axis=0, append=[-100,-100])
    df_labels_list[i] = torch.Tensor(df_labels_list[i].astype('float64'))
    df_data_list[i] = torch.Tensor(df_data_list[i].astype('float64'))

def train_model(model,save_filepath,training_loader,validation_loader,epochs,device):
    
    model.to(device)
    
    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)

    #splitting the dataloaders to generalize code
    data_loaders = {"train": training_loader, "val": validation_loader}

    """
    This is your optimizer. It can be changed but Adam is generally used. 
    Learning rate (alpha in gradient descent) is set to 0.001 but again 
    can easily be adjusted if you are getting issues

    Loss function is set to Mean Squared Error. If you switch to a classifier 
    I'd recommend switching the loss function to nn.CrossEntropyLoss(), but this 
    is also something that can be changed if you feel a better loss function would work
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #loss_func = nn.MSELoss()
    loss_func = nn.CrossEntropyLoss()
    decay_rate = 0.93 #decay the lr each step to 93% of previous lr
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    total_start = time.time()

    """
    You can easily adjust the number of epochs trained here by changing the number in the range
    """
    for epoch in tqdm(range(epochs), position=0, leave=True):
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        temp_loss = 100000000000000.0
        correct = 0
        total = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            for i, (x, y) in enumerate(data_loaders[phase]):  
                x = x.to(device)
                y = y.type(torch.LongTensor).to(device)
                output = model(x)     
#                 print(output)
#                 print(y)
                loss = loss_func(torch.squeeze(output), torch.squeeze(y))
    
                correct += (torch.max(output, 1)[1] == torch.max(y, 1)[1]).float().sum()
                total += list(y.size())[1]
        
                #backprop             
                optimizer.zero_grad()           
                if phase == 'train':
                    loss.backward()
                    optimizer.step()                                      

                #calculating total loss
                running_loss += loss.item()
#                 print(loss.item())
            
            if phase == 'train':
                train_loss = running_loss
                lr_sch.step()
            else:
                val_loss = running_loss

        end = time.time()
        # shows total loss
        if epoch%5 == 0:
#             tqdm.write('accuracy: {} correct: {} total: {}'.format(correct/total, correct, total))
            tqdm.write('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss, val_loss))
#         print(end - start)
        
        #saving best model
        if val_loss < temp_loss:
            torch.save(model, save_filepath)
            temp_loss = val_loss
        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    total_end = time.time()
#     print(total_end - total_start)
    #Creating loss csv
    loss_df = pd.DataFrame(
        {
            'epoch': epochs_list,
            'training loss': train_loss_list,
            'validation loss': val_loss_list
        }
    )
    # Writing loss csv, change path to whatever you want to name it
    loss_df.to_csv(LOSS_PATH, index=None)
    return train_loss_list, val_loss_list

class SeqDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset, _labels):
        self.dataset = _dataset
        self.labels = _labels

    def __getitem__(self, index):
        example = self.dataset[index]
        target = self.labels[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)
    
train_loader = torch.utils.data.DataLoader(dataset=SeqDataset(df_data_list[:40], df_labels_list[:40]),
                                           batch_size=1,
                                           shuffle=False)

validation_loader = torch.utils.data.DataLoader(dataset=SeqDataset(df_data_list[40:], df_labels_list[40:]),
                                           batch_size=1,
                                           shuffle=False)

input_size = 20
hidden_size = 15
output_size = 3
batch_size = 1
num_layers = 1
batch_first = True
dropout = 0.0
epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
h0 = torch.randn(num_layers, batch_size, hidden_size).to(device)
c0 = torch.randn(num_layers, batch_size, hidden_size).to(device)
model = baselineLSTM(input_size, hidden_size, output_size, batch_size, num_layers, batch_first, dropout, h0,c0)

train_loss, validation_loss = train_model(model,MODEL_PATH,train_loader,validation_loader,epochs,device)

def predict_fn(arr):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    arr = arr.to(device)
    
    pred = torch.max(model(arr))[1]
    return pred.detach().cpu().numpy()

feat_names =['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
             'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm',
             'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 
             'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
exp = ltb.RecurrentTabularExplainer(df_data_list[0].reshape(1,-1,20),df_labels_list[0].reshape(1,-1,20),feature_names=feat_names)
explanation = exp.explain_instance(df_data_list[0].reshape(1,-1,20))