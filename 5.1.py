#!/usr/bin/env python
# coding: utf-8

# In[78]:


from mxnet import autograd, nd
from mxnet.gluon import nn


# In[79]:


#二维互相关运算
def corr2d(X, K):  
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# In[80]:


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
corr2d(X, K)


# In[81]:


#二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。
#卷积层的模型参数包括了卷积核和标量偏差
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))
        
    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


# In[82]:


X = nd.ones((6, 8))
X[:, 2:6] = 0
X


# In[83]:


K = nd.array([[11, -10]])
K


# In[84]:


Y = corr2d(X, K)
Y


# In[85]:


# 构造一个输出通道数为1，核数组形状是(1, 2)的二维卷积层
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()


# In[86]:


# 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))


# In[87]:


for i in range(30):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))


# In[88]:


conv2d.weight.data().reshape((1, 2))


# In[ ]:




