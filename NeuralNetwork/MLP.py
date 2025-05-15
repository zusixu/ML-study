"""
使用numpy实现多层感知机
当前计划实现：
1. 前向传播 : yes
2. 反向传播
3. 训练
4. 测试
5. 可视化
6. 自定义激活函数
7. 自定义损失函数
8. 自定义优化器
9. 自定义层数与节点数
"""



import numpy as np

class layer():
    def __init__(self, input_dim, output_dim, activation='sigmoid'):
        """
        input_dim: 输入维度
        output_dim: 输出维度
        activation: 激活函数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim, output_dim) # ，每一列代表一个神经元的权重
        self.b = np.random.randn(1,output_dim)
        self.next = None
        self.pre = None
        try:
            if activation == 'sigmoid':
                self.activation = self.sigmoid
            else:
                print('激活函数未定义')
        except:
            print('激活函数未定义')

    def forward(self, x: np.ndarray):
        """
        parame:
            x: 输入 m行n列，每行代表一个样本
        return:
            y: 输出 m行列，每行代表一个样本
        """
        self.x = x
        self.z = np.matmul(x, self.weight) + self.b
        result = self.activation(self.z)
        return result
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class MLP():
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_nodes_num: list, 
        activation='sigmoid',lsos_function = 'MSE', optimizer = 'SGD'):
        """
        parame:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_layers: 隐藏层层数
            hidden_nodes_num: 每层隐藏层节点数
            activation: 激活函数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.hidden_dim = hidden_layers
        self.hidden_nodes_num = hidden_nodes_num
        self.activation = activation
        
        # 构建网络
        self.input_layer = layer(input_dim, hidden_nodes_num[0], activation)
        cur = self.input_layer
        for i in range(hidden_layers-1):
            cur.next = layer(hidden_nodes_num[i], hidden_nodes_num[i+1], activation)
            cur.next.pre = cur
            cur = cur.next
        self.output_layer = layer(hidden_nodes_num[-1], output_dim, activation)
        self.output_layer.pre = cur
        cur.next = self.output_layer
        
    def sigmoid_prime(self, x):
        """sigmoid函数的导数"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        parame:
            x: 输入
        return:
            y: 输出
        """
        cur = self.input_layer
        value = x
        while cur:
            value = cur.forward(value)
            cur = cur.next
        return value

    def backward(self,x, y):
        """
        parame:
            x: 输入
            y: 输出
        return:
            loss: 损失
        """
        # 前向传播计算激活值和激活前输出
        cur = self.input_layer
        zs = []
        activations = [x]
        deltab = []
        deltaw = []
        value = x 
        while cur:
            value = cur.forward(value)
            zs.append(cur.z)
            activations.append(value)
            cur = cur.next
        activations = [np.array(activation) for activation in activations]
        zs = [np.array(z) for z in zs]
        # 求输出层的误差 
        delta = np.array((activations[-1] - y.reshape(-1,1))) * self.sigmoid_prime(zs[-1])
        
        # 反向传播更新权重
        cur = self.output_layer
        deltab.append(delta)
        deltaw.append(np.matmul(activations[-2].T, delta))
        cur = cur.pre
        index = -2
        while cur:
            delta = np.matmul(cur.next.weight, delta.T).T*self.sigmoid_prime(zs[index])
            deltab.append(delta)
            deltaw.append(np.matmul(activations[index-1].T, delta))
            cur = cur.pre
            index -= 1
        deltaw.reverse()
        deltab.reverse()
        return deltaw, deltab
 
    def train(self, x, y, epochs, batch_size, learning_rate):
        """
        parame:
            x: 输入
            y: 输出
            epochs: 迭代次数
            batch_size: 批大小
            learning_rate: 学习率
        """
        for i in range(epochs):
            for j in range(0, len(x), batch_size):
                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
                deltaw, deltab = self.backward(x_batch, y_batch)
                cur = self.input_layer
                for i in range(len(deltaw)):
                    cur.weight -= learning_rate * deltaw[i]
                    cur.b -= learning_rate * deltab[i].sum(axis=0)
                    cur = cur.next
            print('epoch: ', i, 'loss: ', self.loss(x, y))

    def loss(self, x, y):
        """
        parame:
            x: 输入
            y: 输出
        return:
            loss: 损失
        """
        y_pred = self.forward(x)
        return np.sum((y_pred - y) ** 2) / len(x)

    def predict(self, x):
        """
        parame:
            x: 输入
        return:
            y: 输出
        """
        return self.forward(x)

    def evaluate(self, x, y):
        """
        parame:
            x: 输入
            y: 输出
        return:
            accuracy: 准确率
        """
        y_pred = self.predict(x)
        return np.sum(y_pred == y) / len(x)

if __name__ == '__main__':
    mlp = MLP(2, 1, 1, [3]) 
    # 生成特征数据
    X = np.random.randn(100, 2)
    # 生成目标变量（带有一些非线性特征和噪声）
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.sin(X[:, 0] + X[:, 1]) + np.random.randn(100)
    
    mlp.train(X, y, 1000, 10, 0.1)
    print(mlp.evaluate(X, y))