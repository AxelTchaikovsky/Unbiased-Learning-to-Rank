"""
LambdaRank:
From RankNet to LambdaRank to LambdaMART: An Overview
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf

ListWise Rank
1. For each query's returned document, calculate the score Si, and rank i (forward pass)
    dS / dw is calculated in this step
2. Without explicit define the loss function L, dL / dw_k = Sum_i [(dL / dS_i) * (dS_i / dw_k)]
3. for each document Di, find all other pairs j, calculate lambda:
    for rel(i) > rel(j)
    lambda += - N / (1 + exp(Si - Sj)) * (gain(rel_i) - gain(rel_j)) * |1/log(pos_i+1) - 1/log(pos_j+1)|
    for rel(i) < rel(j)
    lambda += - N / (1 + exp(Sj - Si)) * (gain(rel_i) - gain(rel_j)) * |1/log(pos_i+1) - 1/log(pos_j+1)|
    and lambda is dL / dS_i
4. in the back propagate send lambda backward to update w

to compare with RankNet factorization, the gradient back propagate is:
    pos pairs
    lambda += - 1/(1 + exp(Si - Sj))
    neg pairs
    lambda += 1/(1 + exp(Sj - Si))

to reduce the computation:
    in RankNet
    lambda = sigma * (0.5 * (1 - Sij) - 1 / (1 + exp(sigma *(Si - Sj)))))
    when Rel_i > Rel_j, Sij = 1:
        lambda = -sigma / (1 + exp(sigma(Si - Sj)))
    when Rel_i < Rel_j, Sij = -1:
        lambda = sigma  / (1 + exp(sigma(Sj - Si)))

    in LambdaRank
    lambda = sigma * (0.5 * (1 - Sij) - 1 / (1 + exp(sigma *(Si - Sj))))) * |delta_NDCG|
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from load_mslr import get_time
from metrics import NDCG
from utils import (
    eval_cross_entropy_loss,
    eval_ndcg_at_k,
    get_device,
    get_ckptdir,
    init_weights,
    load_train_vali_data,
    get_args_parser,
    save_to_ckpt,
)
from LambdaRank import LambdaRank

#####################
# test LambdaRank
######################
def test(
    start_epoch=0, additional_epoch=90, lr=0.0001, optim="adam", leaky_relu=False,
    ndcg_gain_in_train="exp2", sigma=1.0,
    double_precision=False, standardize=False,
    small_dataset=False, debug=False,
    output_dir="/tmp/ranking_output/",
):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    writer = SummaryWriter(output_dir)

    precision = torch.float64 if double_precision else torch.float32

    # get training and validation data:
    data_fold = 'Fold1'
    valid_loader, df_valid, test_loader, df_test = load_train_vali_data('Fold1', small_dataset=True)
    print(test_loader.num_features)
    if standardize:
        df_train, scaler = train_loader.train_scaler_and_transform()
        df_valid = valid_loader.apply_scaler(scaler)

    lambdarank_structure = [136, 64, 16]

    net = LambdaRank(lambdarank_structure, leaky_relu=leaky_relu, double_precision=double_precision, sigma=sigma)
    device = get_device()
    net.to(device)
    net.load_state_dict(torch.load("ckptdir\lambdarank-136-64-16-scale-1.0"))
    print(net)

    ckptfile = get_ckptdir('lambdarank', lambdarank_structure, sigma)
    
    net.eval()
    with torch.no_grad():


        count = 0
        batch_size = 200
        grad_batch, y_pred_batch = [], []

        for X, Y in test_loader.generate_batch_per_query():
            X_tensor = torch.tensor(X, dtype=precision, device=device)
            y_pred = net(X_tensor)
            y_pred_batch.append(y_pred)
            # compute the rank order of each document
            rank_df = pd.DataFrame({"Y": y_pred, "doc": np.arange(Y.shape[0])})
            rank_df = rank_df.sort_values("Y").reset_index(drop=True)
            rank_order = rank_df.sort_values("doc").index.values + 1

           


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--sigma", dest="sigma", type=float, default=1.0)
    args = parser.parse_args()

    #didlist = np.load('did.npy')
    did_df = pd.read_table('ranking\\test.init_list',header=None,sep=' ',names=['qid','did1','did2','did3','did4','did5','did6','did7','did8','did9','did10'])
    
    leaky_relu=False
    ndcg_gain_in_train="exp2"
    sigma=1.0
    #writer = SummaryWriter(output_dir)
    double_precision=False
    precision = torch.float64 if double_precision else torch.float32

    # get training and validation data:
    data_fold = 'Fold1'
    valid_loader, df_valid, test_loader, df_test = load_train_vali_data('Fold1', small_dataset=True)
    print(test_loader.num_features)
    qid_list = df_test.loc[:,'qid'].values

    breakpoint

    lambdarank_structure = [136, 64, 16]

    net = LambdaRank(lambdarank_structure, leaky_relu=leaky_relu, double_precision=double_precision, sigma=sigma)
    device = get_device()
    net.to(device)
    net.load_state_dict(torch.load("D:\Data_Mining\pytorch-examples-master\\ranking\ckptdir\\ranknet-factorize-136-64-16"))
    print(net)

    ckptfile = get_ckptdir('lambdarank', lambdarank_structure, sigma)
    
    net.eval()
    with torch.no_grad():


        count = 0
        batch_size = 200
        grad_batch = []
        rank_al_batch = np.array([])
        y_batch  = np.array([])
        qid = 0

        for X, Y in test_loader.generate_batch_per_query():
            X_tensor = torch.tensor(X, dtype=precision, device=device)
            y_pred = net(X_tensor)
            print(len(y_pred))
            #y_pred_batch.append(y_pred)
            y = y_pred.cpu().numpy()
            y = y.reshape(-1)
            y_batch = np.append(y_batch, y)
            #print(y)
            # compute the rank order of each document
            rank_df = pd.DataFrame({"Y": y, "doc": np.arange(y.shape[0])})
            rank_df = rank_df.sort_values("Y").reset_index(drop=True)
           # print(rank_df)
            rank_order = rank_df.sort_values("doc").index.values + 1
           # print(rank_order)

            l = did_df.loc[did_df.qid==qid]
            docs = l.values
            docs = np.delete(docs, 0, axis = 1)
            docs = docs.reshape(-1)[:len(y_pred)]
            print(len(docs.reshape(-1)))
            rank_al = pd.DataFrame({"Rk": rank_order, "doc": docs})
           # print(rank_al)
            rank_al = rank_al.sort_values("Rk",ascending=False)
            rank_al = rank_al.loc[:,'doc'].to_numpy()
            #print(rank_al)
            rank_al_batch = np.append(rank_al_batch, rank_al)
            #print(rank_al_batch)
            
            print(qid)
            qid += 1

        result = np.vstack((qid_list,rank_al_batch.reshape(-1)))
        result_df = pd.DataFrame(result,  index=['QueryID','DocumentID']).astype(int)
        print(result_df)
        result_df = result_df.T
        result_df.to_csv('testout10RN.csv',index=0)
        print(result_df)
        print(result)
        #score_df = pd.DataFrame(y_batch)
        #score_df.to_csv('scores.csv', index = 0)
