# %%
import pandas as pd 
import numpy as np 
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm_notebook
import torch as t 
from model import myNet

# %%

#feature
train_feature = pd.read_csv('data/train/feature.csv')
train_feature = train_feature.to_numpy()

test_feature = pd.read_csv('data/test/feature.csv')
test_feature = test_feature.to_numpy()

valid_feature = pd.read_csv('data/val/feature.csv')
valid_feature = valid_feature.to_numpy()

#click
train_click = pd.read_csv('data_large/train/train.click', header=None, sep=' ', index_col=0)
train_click = train_click.to_numpy()

valid_click = pd.read_csv('data_large/valid/valid.click', header=None, sep=' ', index_col=0)
valid_click = valid_click.to_numpy()
print(valid_click)

train_click[train_click != 1] = -1
valid_click[valid_click != 1] = -1

#init_list
train_init = pd.read_csv('data_large/train/train.init_list', header=None, sep=' ', index_col=0)
train_init = train_init.to_numpy().astype(int)

test_init = pd.read_csv('data_large/test/test.init_list', header=None, sep=' ', index_col=0)
test_init = test_init.to_numpy().astype(int)

valid_init = pd.read_csv('data_large/valid/valid.init_list', header=None, sep=' ', index_col=0)
valid_init = valid_init.to_numpy().astype(int)

train_init[train_init < 0] = train_feature.shape[0]
valid_init[valid_init < 0] = valid_feature.shape[0]
test_init[test_init < 0] = test_feature.shape[0]




# %%

x,y = train_init.shape
train_rel, _ = np.meshgrid(np.arange(1,y+1), np.arange(x))#.astype(np.float64)

x,y = test_init.shape
test_rel, _ = np.meshgrid(np.arange(1,y+1), np.arange(x))#.astype(np.float64)

x,y = valid_init.shape
valid_rel, _ = np.meshgrid(np.arange(1,y+1), np.arange(x))#.astype(np.float64)


# %%
def met_func(click, weight):
    score = np.zeros_like(weight)
    pos = click > 0
    neg = click < 0
    score[pos] = click[pos] / weight[pos]
    score[neg] = click[neg] / weight[neg]
    return score


print(train_click)
print(train_rel)

print(train_click / train_rel)

train_met = met_func(train_click, train_rel)
valid_met = met_func(valid_click, valid_rel)

print(train_met)

train_score = np.zeros(train_feature.shape[0]+1)
valid_score = np.zeros(valid_feature.shape[0]+1)
train_cnt = np.zeros(train_feature.shape[0]+1)
valid_cnt = np.zeros(valid_feature.shape[0]+1)

train_score = np.bincount(train_init.reshape(-1), weights=train_met.reshape(-1))
print(train_met.reshape(-1))
valid_score = np.bincount(valid_init.reshape(-1), weights=valid_met.reshape(-1))

idx,cnt = np.unique(train_init, return_counts=True)
train_cnt[idx]=cnt
train_score[train_cnt>0] /= train_cnt[train_cnt>0]

idx,cnt = np.unique(valid_init, return_counts=True)
valid_cnt[idx]=cnt
valid_score[valid_cnt>0] /= valid_cnt[valid_cnt>0]

train_score = np.delete(train_score, -1)
valid_score = np.delete(valid_score, -1)
train_cnt = np.delete(train_cnt, -1)
valid_cnt = np.delete(valid_cnt, -1)

print(train_score.shape[0])
print(train_cnt.shape[0])

train_score_val = train_score[train_cnt>0]
train_feature_val = train_feature[train_cnt>0]

print(train_score)

valid_score_val = valid_score[valid_cnt > 0]
valid_feature_val = valid_feature[valid_cnt > 0]


# %%
train_loader = data.DataLoader(
    data.TensorDataset(
        t.from_numpy(train_feature_val).to(dtype=t.float32),
        t.from_numpy(train_score_val).to(dtype=t.float32)),
        batch_size = 20000,
        shuffle = True
)

valid_loader = data.DataLoader(
    data.TensorDataset(
        t.from_numpy(valid_feature_val).to(dtype=t.float32),
        t.from_numpy(valid_score_val).to(dtype=t.float32)),
        batch_size = 20000,
        shuffle = False
)



# %%
#nn_structure = [136, 64, 16]
cuda=t.device(0)
#net = myNet(nn_structure).to(cuda)

net = myNet(train_feature.shape[1], 3 * train_feature.shape[1], 2, nn.LeakyReLU, 0.4).to(cuda)

#net = myNet().to(cuda)
criterion = nn.MSELoss()
weight_decay = 0.01
optimizer = t.optim.Adam(net.parameters(),lr=1e-4, weight_decay=weight_decay)
NUM_EPOCHS = 150

loss_train = np.zeros(NUM_EPOCHS)
loss_valid = np.zeros(NUM_EPOCHS)


for epoch in tqdm_notebook(range(NUM_EPOCHS)):
    net.train()
    for feature, score in train_loader:
        #data is a batch of featuresets and labels
        feature = feature.to(cuda)
        score = score.to(cuda)
        predict = net(feature)
        net.zero_grad()
        loss = criterion(predict, score)
        loss.backward()
        optimizer.step()
        loss_train[epoch]+=loss.item()*feature.shape[0]
    loss_train[epoch]/=len(train_loader.dataset)
    #print(len(train_loader.dataset))

    net.eval()
    with t.no_grad():
        for feature, score in valid_loader:
            feature = feature.to(cuda)
            score = score.to(cuda)
            predict = net(feature)
            loss = criterion(predict, score)
            loss_valid[epoch] += loss.item()*feature.shape[0]
        loss_valid[epoch]/=len(valid_loader.dataset)
        #print(len(valid_loader.dataset))

# %%
plt.plot(loss_train, label='train', c="C6")
plt.plot(loss_valid, label='validation', c="C9")
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.savefig('pic.png', dpi=150)
plt.show()

# %%
test_loader = data.DataLoader(data.TensorDataset(t.from_numpy(test_feature).to(dtype=t.float32)), batch_size=20000, shuffle=False)

net.eval()
test_scores = np.array([])

with t.no_grad():
    for [feature] in test_loader:
        feature = feature.to(cuda)
        score = net(feature)
        test_scores = np.concatenate([test_scores, score.cpu().numpy()])
test_scores = np.append(test_scores, -np.inf)
test_bias = test_scores[test_init]
test_order = np.take_along_axis(test_init, np.argsort(-test_bias), axis=1)
_, queryId = np.meshgrid(np.arange(test_order.shape[1]), np.arange(test_order.shape[0]))
result = np.vstack((queryId.reshape(-1), test_order.reshape(-1)))
result = np.delete(result, np.argwhere(result[1, :] == test_order.max()), axis=1)
result = result.T

df = pd.DataFrame(result, columns=['QueryId', 'DocumentId'])
df.to_csv('result/06044.csv', index=False)

# %%
