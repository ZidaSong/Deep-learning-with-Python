#!/usr/bin/env python
# coding: utf-8

# In[15]:


from mxnet import autograd,nd

num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]
labels+=nd.random.normal(scale=0.01,shape=labels.shape)


# In[16]:


from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)


# In[17]:


for X, y in data_iter:
    print(X, y)
    break


# In[18]:


from mxnet.gluon import nn

net = nn.Sequential()


# In[19]:


net.add(nn.Dense(1))


# In[20]:


from mxnet import init

net.initialize(init.Normal(sigma=0.01))


# In[21]:


from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()


# In[22]:


from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})


# In[23]:


num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))


# In[24]:


dense = net[0]
true_w, dense.weight.data()


# In[25]:


true_b, dense.bias.data()


# In[ ]:




