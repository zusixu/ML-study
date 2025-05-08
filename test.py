import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
# read data
from sklearn.model_selection import train_test_split

#%%

# 生成包含1000个样本的随机数据集
np.random.seed(0)
X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

#%%


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# create model instance
parameters = {
    "objective": "binary:logistic",
    # "booster": "gblinear",
    # "device": "cuda",
    "verbosity": "2",
    "eta": "0.5",
    # "gamma": "0.1",
    "max_depth": "5",
    "min_child_weight": "1",
    "subsample": "1",
    # "lambda": "0.1",
    # "alpha": "0.1",
    # "updater": "shotgun",
    "tree_method": "exact",
    # "sampling_method": "gradient_based"
}

watchlist = [(dtrain, "train"), (dtest, "test")]
model = xgb.train(parameters, dtrain, num_boost_round=500, evals=watchlist)
#%%
result = np.around(model.predict(dtest), 2)
#%%
p_point =[]
rec_point = []
thresholds = np.arange(0, 1, 0.01)
for threshold in thresholds:
    y_pred = (result >= threshold).astype(int)
    TP = np.sum((y_pred == 1) & (y_test == 1))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))

    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    p_point.append(precision)
    rec_point.append(recall)

plt.plot(rec_point, p_point)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P-R Curve')
plt.grid(True)
plt.show()
