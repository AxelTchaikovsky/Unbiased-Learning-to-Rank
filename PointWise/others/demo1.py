import pandas as pd
import numpy as np
origin_rank=pd.read_table('data_large/train/train.init_list',header=None,sep=' ',names=['qid','did1','did2','did3','did4','did5','did6','did7','did8','did9','did10'])
print(origin_rank)
nn=['did']
for i in range(1,136):
    tmp='f'+str(i)
    nn.append(tmp)
print(nn)
train_feature=pd.read_table('data_large/train/train.feature',header=None,sep=' ',names=nn)
train_click=pd.read_table('data_large/train/train.click',header=None,sep=' ',names=['qid','click1','click2','click3','click4','click5','click6','click7','click8','click9','click10'])

