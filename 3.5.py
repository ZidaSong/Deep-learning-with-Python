#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time


# In[2]:


mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)


# In[3]:


len(mnist_train), len(mnist_test)


# In[7]:


feature, label = mnist_train[2]


# In[8]:


feature.shape, feature.dtype


# In[9]:


label, type(label), label.dtype


# In[10]:


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# In[11]:


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


# In[12]:


X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))


# In[13]:


batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)


# In[14]:


start = time.time()
for X, y in train_iter:
    continue
'%.2f sec' % (time.time() - start)


# In[ ]:




