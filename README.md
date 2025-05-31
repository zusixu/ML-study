# Introduction
This is a project that reproduces common machine learning algorithms for learning machine learning. In the example.py file under each folder, the code implementation using the sklearn library directly is also provided. In the future, I will learn how to build these algorithms by sklearn.base, I think it will be a better way to learn how to build machine learning algorithms.
# Environment
```
pip install requirements.txt
```
# Index
Here is the list of algorithms that have been implemented:
- Linear Model
    - [Logistic Regression and Linear Regression](https://github.com/zusixu/Machine-Learing/blob/main/LinearModel/LR.py)
    - [LDA](https://github.com/zusixu/Machine-Learing/blob/main/LinearModel/lda.py)
- Decision Tree
    - [C4.5](https://github.com/zusixu/Machine-Learing/blob/main/DecisionTree/C45.py)
    - [CART](https://github.com/zusixu/Machine-Learing/blob/main/DecisionTree/CART.py)
        - I find a bug in CART.py. When there are missing values for a certain attribute in the dataset to be split, and other attributes cannot be used for splitting based on the Gini index, the current approach is to assign the missing values to all subtrees and update the count weights. However, this leads to poor performance on data.csv. I have not yet figured out how to solve this problem.
- Neural Network
    - [MLP(BP)](https://github.com/zusixu/Machine-Learing/blob/main/NeuralNetwork/MLP.py): It's intersting to achieve a mlp model with only numpy. In the future, I will try to use numpy to implement some functions of pytorch. [This](https://www.cnblogs.com/pinard/p/6422831.html#) is a very good reference, which provides detailed formula derivations.
    - [RNN](https://github.com/zusixu/Machine-Learing/blob/main/NeuralNetwork/RNN.py) I have already create the file, and I will finish it (Maybe)/doge  :)

- Cluster
    - [K-Means](https://github.com/zusixu/Machine-Learing/blob/main/Cluster/KMeans.py) This is my first time know [broadcast](https://www.runoob.com/numpy/numpy-broadcast.html) in numpy, I think it's very useful and interesting. np.newaxis is helpful too. WKMeans is a better version of KMeans, but I haven't figured out the [formula derivation](https://zhuanlan.zhihu.com/p/157106355) yet, so I will implement it in the future.