#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss


# In[2]:


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# In[3]:


num_inputs, num_outputs, num_hiddens = 784, 10, 256
#设隐藏单元个数为256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


# In[4]:


def relu(X):
    return nd.maximum(X, 0)


# In[6]:


#定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2


# In[7]:


#损失函数
loss = gloss.SoftmaxCrossEntropyLoss()


# In[8]:


num_epochs, lr = 5, 0.5
#调用了3.6中的函数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)


# In[9]:


for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])


# In[ ]:




