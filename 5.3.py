#!/usr/bin/env python
# coding: utf-8

# In[1]:


import d2lzh as d2l
from mxnet import nd

def corr2d_multi_in(X, K):
    # 首先沿着X和K的第0维（通道维）遍历。然后使用*将结果列表变成add_n函数的位置参数来进行相加
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])


# In[2]:


X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

corr2d_multi_in(X, K)


# In[3]:


def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])


# In[4]:


K = nd.stack(K, K + 1, K + 2)
K.shape


# In[5]:


corr2d_multi_in_out(X, K)


# In[6]:


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)  # 全连接层的矩阵乘法
    return Y.reshape((c_o, h, w))


# In[7]:


X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

(Y1 - Y2).norm().asscalar() < 1e-6


# In[ ]:




