#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


# In[2]:


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
pool2d(X, (2, 2))


# In[3]:


pool2d(X, (2, 2), 'avg')


# In[4]:


X = nd.arange(16).reshape((1, 1, 4, 4))
X


# In[5]:


pool2d = nn.MaxPool2D(3)
pool2d(X)  # 因为池化层没有模型参数，所以不需要调用参数初始化函数


# In[6]:


pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)


# In[7]:


pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)


# In[8]:


X = nd.concat(X, X + 1, dim=1)
X


# In[9]:


pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)


# In[ ]:




