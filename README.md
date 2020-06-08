# EE448: Unbiased-Learning-to-Rank

A Kaggle project for search engine ranking task utilizing several state-of-the-art ranking algorithms.

### Point-wise

Feed forward NN, minimize document pointwise MSELoss function. 

to train the model

```
python PointWise/pointwise.py 
```

for more, click in and run the interactive modules.

### RankNet

Feed forward NN, minimize document pairwise cross entropy loss function

to train the model

```
python RkN&LaRk/RankNet.py --lr 0.0001 --debug --standardize
```

`--debug` print the parameter norm and parameter grad norm. This enable to evaluate whether there is gradient vanishing and gradient exploding problem
`--standardize` makes sure input are scaled to have 0 as mean and 1.0 as standard deviation

NN structure: 136 -> 64 -> 16 -> 1, ReLU6 as activation function

### LambdaRank

Feed forward NN. Gradient is proportional to NDCG change of swapping two pairs of document

to choose the optimal learning rate, use smaller dataset:

```
python RkN&LaRk/LambdaRank.py --lr 0.001 --ndcg_gain_in_train exp2 --small_dataset --debug --standardize
```

otherwise, use normal dataset:

```
OUTPUT_DIR=/tmp/ranking_output/
python ranking/LambdaRank.py --lr 0.01 --ndcg_gain_in_train exp2 --standardize \
--output_dir=$OUTPUT_DIR
```

to switch identity gain in NDCG in training, use `--ndcg_gain_in_train identity`

For output, run RkN&LaRk/RankNet.py which will generate a formatted .csv result. 

## Dependencies:

* pytorch-1.0
* pandas
* numpy
* sklearn