# help function
from transfer_learning import NeuralNet
#from dataset_loader import data_loader, all_filter, get_descriptors, one_filter, data_scaler

# modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import os, sys
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing

from tqdm import tqdm
from scipy.stats import pearsonr

import matplotlib.pyplot as plt 
#%matplotlib inline

# file name and data path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = os.getcwd()
#file_name = 'CrystGrowthDesign_SI.csv'
print (device)
print ("Bingo!")


class SequenceDataset(Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)


X = np.load("../chain_double_clean.npy")
y = np.load("../F_double_clean.npy")
#X = np.load("../../../SEQ_PPI_24_16/chain_L24_maxmin.npy")
#y = np.load("../../../SEQ_PPI_24_16/PP_F_L24_maxmin.npy")
X = X[0:20000]
y = y[0:20000]

#y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(20000,)

rnumbers = np.load("../../randomlist.npy")
rnumbers = rnumbers[0:1000]

test_scores = []

for rnumber in rnumbers:
	

	X1, X2, y1, y2 = train_test_split(X, y, test_size=0.01, random_state=rnumber)

	y2 = preprocessing.StandardScaler().fit_transform(y2.reshape(-1, 1)).reshape(200,)

	X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.1, random_state=1)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

	#print ("whole small data size:",X2.shape)
	#print ("train size", X_train.shape)
	#print ("test size:", X_test.shape)
	#print ("val size:", X_val.shape)

	# 把 data 做成 dataset 供 dataloader 取用
	train_dataset = SequenceDataset(X=X_train, y=y_train)
	val_dataset = SequenceDataset(X=X_val, y=y_val)
	test_dataset = SequenceDataset(X=X_test, y=y_test)

	batch_size = 32
	# 把 data 轉成 batch of tensors
	train_loader = DataLoader(dataset = train_dataset,
                                    batch_size = batch_size,
                                    shuffle = True)

	val_loader = DataLoader(dataset = val_dataset,
                                  batch_size = batch_size,
                                  shuffle = False)

	test_loader = DataLoader(dataset = test_dataset,
                                  batch_size = batch_size,
                                  shuffle = False)

	## hyper-parameters
	input_size = 20
	hidden_size_1 = 64
	hidden_size_2 = 64
	hidden_size_3 = 32
	output_size = 1
	learning_rate = 0.00002

	## model, loss, and optimizer
	model = NeuralNet(input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size).to(device)
	criterion = nn.MSELoss()
	#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	checkpoint = torch.load("checkpoint_PS1_best.pt")
	model.load_state_dict(checkpoint['model_state_dict'])
	model.fc1.weight.requires_grad = False
	model.fc1.bias.requires_grad = False
	#model.fc2.weight.requires_grad = False
	#model.fc2.bias.requires_grad = False
	#model.fc3.weight.requires_grad = False
	#model.fc3.bias.requires_grad = False
	#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



	t_batch = len(train_loader) 
	v_batch = len(val_loader)

	num_epochs = 10000
	scores_epochs_train = list()
	scores_epochs_val = list()
	losses_epochs_train = list()
	losses_epochs_val = list()
	best_score = -10.0

	for epoch in range(num_epochs):
    	#epoch_start_time = time.time()
    	#train_acc = 0.0
	    train_loss = 0.0
	    #val_acc = 0.0
	    val_loss = 0.0

	    # 這段做 training
	    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
	    for i, (inputs, labels) in enumerate(train_loader):
	        inputs = inputs.to(device, dtype=torch.float)
	        labels = labels.to(device, dtype=torch.float)
	        optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
	        outputs = model(inputs) # 將 input 餵給模型
	        outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
	        loss = criterion(outputs, labels) # 計算此時模型的 training loss
	        loss.backward() # 算 loss 的 gradient
	        optimizer.step() # 更新訓練模型的參數
	        train_loss += loss.item()
	        #print('[ Epoch{}: {}/{} ] loss:{:.3f}'.format(epoch+1, i+1, t_batch, loss.item(), end='\r'))
	    #print('\nTrain | Loss:{:.5f}'.format(train_loss/t_batch))
	    losses_epochs_train.append(train_loss/t_batch)
	    # 這段做 validation
	    model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
	    with torch.no_grad():
	        total_loss, total_acc = 0, 0
	        for i, (inputs, labels) in enumerate(val_loader):
	            inputs = inputs.to(device, dtype=torch.float) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
	            labels = labels.to(device, dtype=torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
	            outputs = model(inputs) # 將 input 餵給模型
	            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
	            loss = criterion(outputs, labels) # 計算此時模型的 validation loss
	            val_loss += loss.item()
	        #print("Valid | Loss:{:.5f} ".format(val_loss/v_batch))
	        losses_epochs_val.append(val_loss/v_batch)
    
	    inputs_train = torch.from_numpy(X_train)
	    labels_train = torch.from_numpy(y_train)
	    outputs_train = model(inputs_train.float()).view(-1,)
	    score_train = r2_score(labels_train.data.numpy(), outputs_train.data.numpy())
	    scores_epochs_train.append(score_train)

	    inputs_val = torch.from_numpy(X_val)
	    labels_val = torch.from_numpy(y_val)
	    outputs_val = model(inputs_val.float()).view(-1,)
	    score_val = r2_score(labels_val.data.numpy(), outputs_val.data.numpy())
	    scores_epochs_val.append(score_val)
	    if best_score < score_val:
	        best_score = score_val
	        torch.save({
	            'epoch': epoch,
	            'model_state_dict': model.state_dict(),
	            'optimizer_state_dict': optimizer.state_dict(),
	            'loss': loss
	            }, "checkpoint_best.pt")
	        #plt.figure(figsize=(6,6))
	        #plt.scatter(labels_train.data.numpy(), outputs_train.data.numpy(), c='blue')
	        #plt.scatter(labels_val.data.numpy(), outputs_val.data.numpy(), c='red')
	        #plt.savefig("Performance_Best.png")
	    length_score = len(scores_epochs_val)
	    if length_score >=200 :
	        if score_val < scores_epochs_val[length_score-5] and score_val < scores_epochs_val[length_score-20] and score_val < scores_epochs_val[length_score-40] and score_val < scores_epochs_val[length_score-60]:
	            break


	model_best = NeuralNet(input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size).to(device)
	checkpoint_best = torch.load("checkpoint_best.pt")
	model_best.load_state_dict(checkpoint_best['model_state_dict'])

	inputs_test = torch.from_numpy(X_test)
	labels_test = torch.from_numpy(y_test)
	outputs_test = model_best(inputs_test.float()).view(-1,)
	score_test = r2_score(labels_test.data.numpy(), outputs_test.data.numpy())
	print("Test Score:", score_test)
	test_scores.append(score_test)
	np.save("score_test.npy", np.array(test_scores))


np.save("score_test.npy", np.array(test_scores))
plt.figure(figsize=(6,6))
plt.plot(test_scores, c='red', label = "Test")
plt.ylim(0,1)
plt.legend()
plt.savefig("Scores.png")
