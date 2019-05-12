#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import image


# In[3]:


d2l.set_figsize()
img = image.imread('C:/catdog.jpg').asnumpy()
d2l.plt.imshow(img);  # 加分号只显示图


# In[15]:


# bbox是bounding box的缩写
dog_bbox, cat_bbox = [20, 10, 137, 180], [143, 45, 230, 175]


# In[5]:


def bbox_to_rect(bbox, color):  
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


# In[16]:


fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));


# In[ ]:




