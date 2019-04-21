#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')


# In[3]:


x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyplot(x, y, 'relu')


# In[4]:


y.backward()
xyplot(x,x.grad,'grad of relu')


# In[5]:


with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')


# In[6]:


y.backward()
xyplot(x, x.grad, 'grad of sigmoid')


# In[7]:


with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')


# In[ ]:




