# Introduction
This is a project that reproduces common machine learning algorithms for learning machine learning.
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
    - [MLP(BP)](https://github.com/zusixu/Machine-Learing/blob/main/NeuralNetwork/MLP.py): It's intersting to achieve a mlp model with only numpy. In the future, I will try to use numpy to implement some functions of pytorch.