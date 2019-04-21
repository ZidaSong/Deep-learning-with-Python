#!/usr/bin/env python
# coding: utf-8

# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import autograd, nd


# In[24]:


batch_size = 256
#批量大小为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# In[25]:


num_inputs = 784
#输入28*28=784像素
num_outputs = 10
#输出分类种类10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)


# In[26]:


W.attach_grad()
b.attach_grad()
#为模型参数附上梯度


# In[27]:


X = nd.array([[1, 2, 3], [4, 5, 6]])
X.sum(axis=0, keepdims=True), X.sum(axis=1, keepdims=True)
#对列求和（axis=0），对行求和（axis=1），保留行和列两个维度（keepdims=True）


# In[28]:


def softmax(X):
    X_exp = X.exp()
    #对每个元素的概率做指数运算，得出正数
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition


# In[ ]:





# In[29]:


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
#将原始输出图像reshape为长度为num_inputs的向量，并进行了*w+b运算，并softmax计算了gailv


# In[30]:


#pick函数
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([2, 2], dtype='int32')
nd.pick(y_hat, y)


# In[31]:


help(nd.pick)


# In[32]:


#定义了交叉熵损失函数
def cross_entropy(y_hat,y):
    return - nd.pick(y_hat,y).log()


# In[33]:


#定义分类准确率函数
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


# In[34]:


accuracy(y_hat, y)


# In[35]:


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


# In[36]:


evaluate_accuracy(test_iter, net)


# In[41]:


num_epochs,lr=10,0.1

def train_ch3(net,train_iter,test_iter,loss,num_epochs, batch_size, params= None, lr= None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
         [W,b], lr)


# In[42]:


for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])


# In[ ]:




