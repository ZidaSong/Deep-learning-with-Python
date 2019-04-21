#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn


# In[2]:


n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05


# In[8]:


true_w,true_b


# In[5]:


features = nd.random.normal(shape=(n_train + n_test, num_inputs))
features


# In[7]:


labels = nd.dot(features, true_w) + true_b
labels


# In[10]:


labels += nd.random.normal(scale=0.01, shape=labels.shape)
labels


# In[15]:


train_features, test_features = features[:n_train, :], features[n_train:, :]
#features中的冒号一个在前一个在后？
train_features, test_features


# In[22]:


def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]


# In[23]:


#定义L2范数惩罚项
def l2_penalty(w):
    return (w**2).sum() / 2


# In[24]:


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    train_features, train_labels), batch_size, shuffle=True)


# In[25]:


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # 添加了L2范数惩罚项
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().asscalar())


# In[26]:


#过拟合
fit_and_plot(lambd=0)


# In[27]:


#使用权重衰减
fit_and_plot(lambd=3)


# In[ ]:




