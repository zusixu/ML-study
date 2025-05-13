"""
使用numpy实现多层感知机
当前计划实现：
1. 前向传播
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

class node():
    def __init__(self, input_dim, output_dim, activation='sigmoid'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(output_dim)

        self.next = []
        self.pre = []


class MLP():
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_nodes_num: list, activation='sigmoid'):
        """
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_dim: 隐藏层维度
        hidden_nodes_num: 隐藏层节点数
        activation: 激活函数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_nodes_num = hidden_nodes_num
        self.activation = activation
        