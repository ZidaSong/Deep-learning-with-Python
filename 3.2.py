#!/usr/bin/env python
# coding: utf-8

# In[17]:


from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random


# In[18]:


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)


# In[19]:


features[0], labels[0]


# In[20]:


def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);  


# In[21]:


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) 
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


# In[22]:


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break


# In[23]:


w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))


# In[24]:


w.attach_grad()
b.attach_grad()


# In[25]:


def linreg(X, w, b):  
    return nd.dot(X, w) + b


# In[26]:


def squared_loss(y_hat, y): 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# In[27]:


def sgd(params, lr, batch_size): 
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# In[28]:


lr = 0.01
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  
        l.backward() 
        sgd([w, b], lr, batch_size)  
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))


# In[29]:


true_w, w


# In[30]:


true_b, b


# In[31]:


#以下为练习代码
x=[[1,2]]
w=[[3],[4]]


# In[ ]:





# In[32]:


features = nd.random.normal(scale=1, shape=(20, 3))


# In[33]:


features


# In[34]:


help(nd.dot)


# In[35]:


help(nd.zeros)


# In[36]:


help(nd.random.normal)


# In[37]:


help(display.set_matplotlib_formats)


# In[38]:


help(plt.rcParams)


# In[39]:


help(plt.scatter)


# In[ ]:




