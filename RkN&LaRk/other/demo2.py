import numpy as np
# from LambdaRankNN import LambdaRankNN, RankNetNN
# from rankerNN2pmml import rankerNN2pmml
import pandas as pd

def fun(p):
    x=p[p.find(':')+1:len(p)]
    return(float(x))
train_click=pd.read_table('data_large/train/train.click',header=None,sep=' ',names=['qid','click1','click2','click3','click4','click5','click6','click7','click8','click9','click10'])
train_rank=pd.read_table('data_large/train/train.init_list',header=None,sep=' ',names=['qid','did1','did2','did3','did4','did5','did6','did7','did8','did9','did10'])
# train_feature=pd.read_csv('Zfeature.csv')
nn=['did']
for i in range(1,137):
    tmp='f'+str(i)
    nn.append(tmp)
train_feature=pd.read_table('data_large/train/train.feature',header=None,sep=' ',names=nn)
nn.remove('did')
test_rank=pd.read_table('data_large/test/test.init_list',header=None,sep=' ',names=['qid','did1','did2','did3','did4','did5','did6','did7','did8','did9','did10'])
test_feature=pd.read_csv('test_feature.csv')

# nn=[]
# for i in range(1,137):
#     tmp='f'+str(i)
#     nn.append(tmp)
# X=train_feature.loc[:,nn].values


qid=[]
tmp=[]
y=[]
for i in range(10*len(test_rank)):
    a=i//10
    b=i%10
    did=test_rank.loc[a,'did'+str(b+1)]
    if did==did:
        tmp.append(int(did))
        qid.append(['qid:'+str(a)])
        #click=train_click.loc[a,'click'+str(b+1)]
        click  = 0
        y.append([int(click)])
X=test_feature.loc[tmp].values
y=np.array(y)
qid=np.array(qid)
X=np.delete(X,0,axis=1)
data=np.column_stack((y,qid,X))
#np.savetxt('test.txt',data, fmt='%s')#data就是最后的结果
print(tmp)

print(data)
# def score2rank(p,did):
#     pc=p.copy()
#     q=sorted(p,reverse=True)
#     l=[0]*len(q)
#     for i in range(len(q)):
#         itm=q[i]
#         for j in range(len(pc)):
#             if pc[j]==itm:
#                 pc[j]=123
#                 l[i]=name[j]
#                 break;
#     return(l)

# qid=[]
# tmp=[]
# y=[]
# for i in range(10*len(test_rank)):
#     a=i//10
#     b=i%10
#     did=test_rank.loc[a,'did'+str(b+1)]
#     if did==did:
#         tmp.append(int(did))
#         qid.append([a])
#         y.append([0])
# X=train_feature.loc[tmp].values
# y=np.array(y)
# qid=np.array(qid)
# X=np.delete(X,0,axis=1)
# data=np.column_stack((y,qid,X))
# np.save('Z.npy',data)

# idd=0
# tmpp=0
# p=np.load('P.npy')
# rank=[]
# for i in range(len(qid)):
#     id=qid[i][0]
#     if (id!=idd):
#         s=p[tmpp:i]
#         name=tmp[tmpp:i]
#         rank.append(score2rank(s,name))
#         tmpp=i
#         idd=id
#     if (i==len(qid)-1):
#         s=p[tmpp:i+1]
#         name=tmp[tmpp:i+1]
#         rank.append(score2rank(s,name))
#         tmpp=i
#         idd=id

# out=[]
# for i in range(len(rank)):
#     for did in rank[i]:
#         out.append([i,int(did)])
# df = pd.DataFrame(out, columns=['QueryId','DocumentId'])
# df.to_csv('testout.csv',index=0)





# # # generate query data
# # X = np.array([[0.2, 0.3, 0.4],
# #               [0.1, 0.7, 0.4],
# #               [0.3, 0.4, 0.1],
# #               [0.8, 0.4, 0.3],
# #               [0.9, 0.35, 0.25]])
# # y = np.array([0, 1, 0, 0, 2])
# # qid = np.array([1, 1, 1, 2, 2])

# # tmp=[]
# # for i in range(10*len(test_rank)):
# #     a=i//10
# #     b=i%10
# #     did=test_rank.loc[a,'did'+str(b+1)]
# #     if did==did:
# #         tmp.append(int(did))
# #         qid.append(a)
# #         y.append(10-b)

# ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(128,32,), activation=('sigmoid', 'sigmoid'), solver='adam')
# ranker.fit(X, y, qid, epochs=5)
# print(y)
# ranker.evaluate(X, y, qid, eval_at=10)
# # y_pred = ranker.predict(X)
# # print(y_pred)

# qid=[]
# tmp=[]
# for i in range(10*len(test_rank)):
#     a=i//10
#     b=i%10
#     did=test_rank.loc[a,'did'+str(b+1)]
#     if did==did:
#         tmp.append(int(did))
#         qid.append(a)
# # test_X=test_feature.loc[tmp].values

# t=0
# tmp=qid[0]
# # for i in range(1,len(train_feature.qid)):
# for i in range(1,len(qid)):
#     id=qid[i]
#     if id!=tmp:
#         x=X[t:i-1]
#         t=i
#         tmp=id
#         y_pred=ranker.predict(x)
#         print(y_pred)
       
    

# params = {
#     'feature_names': ['Feature1', 'Feature2', 'Feature3'],
#     'target_name': 'score'
# }

# rankerNN2pmml(estimator=ranker.model, file='Model_example.xml', **params)
