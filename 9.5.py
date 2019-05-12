#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import contrib, image, nd

img = image.imread('C:/catdog.jpg')
h, w = img.shape[0:2]
h, w


# In[2]:


d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h))  # 前两维的取值不影响输出结果
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)


# In[3]:


display_anchors(fmap_w=4, fmap_h=4, s=[0.15])


# In[4]:


display_anchors(fmap_w=2, fmap_h=2, s=[0.4])


# In[5]:


display_anchors(fmap_w=1, fmap_h=1, s=[0.8])


# In[ ]:




