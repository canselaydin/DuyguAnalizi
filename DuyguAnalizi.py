#!/usr/bin/env python
# coding: utf-8

# In[7]:



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[8]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[16]:


get_ipython().system('pip install tensorflow')


# In[10]:


import re
import string
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.layers import Embedding, GRU, Dense
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# In[21]:


data=pd.read_csv('C:/Users/canse/OneDrive/Masaüstü/turkish_movie_sentiment_dataset.csv')


# In[22]:


data.head(5)


# In[23]:


data.shape


# # VERİNİN GÖRSELLEŞTİRİLMESİ

# Film puanları dağılımlarının yüzdelik dilimini görmek için Matplotlib kütüphanesini kullanabiliriz.

# In[24]:


plot_size = plt.rcParams["figure.figsize"]
print(plot_size[0])
print(plot_size[1])
plot_size[0] = 10
plot_size[1] = 15
plt.rcParams["figure.figsize"] = plot_size
data['point'].value_counts().plot(kind='pie', autopct='%1.0f%%')


# Burada film puanlarının dağılımını görüyoruz.Sayı bazlı sonuçları görmek için de Seaborn kütüphanesini kullanıyoruz.

# In[25]:


sns.countplot(data['point'])


# # VERİNİN TEMİZLENMESİ

# In[26]:


comments = lambda x : x[23:-24]

data["comment"] = data["comment"].apply(comments)
data["comment"].head()


# In[27]:


floatize = lambda x : float(x[0:-2])

data["point"] = data["point"].apply(floatize)
data["point"].value_counts()


# In[28]:


data.drop(data[data["point"] == 3].index, inplace = True)
data["point"] = data["point"].replace(1, 0)
data["point"] = data["point"].replace(2, 0)
data["point"] = data["point"].replace(4, 1)
data["point"] = data["point"].replace(5, 1)
data["point"].value_counts()


# In[29]:


data.reset_index(inplace = True)
data.drop("index", axis = 1, inplace = True)
data.head()


# In[30]:


data["comment"] = data["comment"].apply(lambda x: x.lower())
data.head()


# In[31]:


def remove_punctuation(text):
    no_punc = [words for words in text if words not in string.punctuation]
    word_wo_punc = "".join(no_punc)
    return word_wo_punc

data["comment"] = data["comment"].apply(lambda x: remove_punctuation(x))
data["comment"] = data["comment"].apply(lambda x: x.replace("\r", " "))
data["comment"] = data["comment"].apply(lambda x: x.replace("\n", " "))

data.head()


# In[32]:


def remove_numeric(corpus):
    output = "".join(words for words in corpus if not words.isdigit())
    return output

data["comment"] = data["comment"].apply(lambda x: remove_numeric(x)) 
data.head()


# In[33]:


target = data["point"].values.tolist()
data = data["comment"].values.tolist()

cutoff = int(len(data)*0.80)

X_train, X_test = data[:cutoff], data[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:]


# In[34]:


num_words = 10000
tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(data)


# In[35]:


X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

print([X_train[1000]])
print(X_train_tokens[1000])


# In[36]:


num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]
num_tokens = np.array(num_tokens)
num_tokens


# In[37]:


np.mean(num_tokens)


# In[38]:


np.max(num_tokens)


# In[39]:


max_tokens = np.mean(num_tokens) + (2*np.std(num_tokens))
max_tokens = int(max_tokens)
max_tokens


# In[40]:


np.sum(num_tokens < max_tokens) / len(num_tokens)


# In[41]:


X_train_pad = pad_sequences(X_train_tokens, maxlen = max_tokens) 
X_test_pad = pad_sequences(X_test_tokens, maxlen = max_tokens)

print(X_train_pad.shape)
print(X_test_pad.shape)


# In[42]:


np.array(X_train_tokens[800])


# In[43]:


X_train_pad[2000]


# In[44]:


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token != 0]
    text = " ".join(words) # Kelimeler aralarında boşluk bırakılarak ard arda yazılacaktır.
    return text


# In[45]:


tokens_to_string(X_train_tokens[350])


# # MODEL OLUŞTURMA

# In[34]:


embedding_size = 50
model = Sequential()
model.add(Embedding(input_dim = num_words, output_dim = embedding_size, input_length = max_tokens, name = "embedding_layer"))
model.add(GRU(units = 16, return_sequences = True))
model.add(GRU(units = 8, return_sequences = True))
model.add(GRU(units = 4))
model.add(Dense(1, activation = "sigmoid"))


# In[35]:


optimizer = Adam(lr = 1e-3)
model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
model.summary()


# In[36]:


X_train_pad = np.array(X_train_pad)
y_train = np.array(y_train)

model.fit(X_train_pad, y_train, epochs = 100, batch_size = 256)
model.save("OneDrive/Masaüstü/duygu_analizi_model.h5")


# In[19]:



from tensorflow.keras import models,layers
model=tf.keras.models.load_model("OneDrive/Masaüstü/duygu_analizi_model.h5")


# # TAHMİN

# In[46]:


y_pred = model.predict(X_test_pad[0:1000])
y_pred = y_pred.T[0]


# In[47]:


cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
cls_true = np.array(y_test[0:1000])


# In[48]:


incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]


# In[49]:


len(incorrect)


# In[103]:


idx = incorrect[6-8]
X_test[idx]


# In[104]:


y_pred[idx]


# In[105]:


cls_true[idx]


# In[109]:


X_test[5]


# In[110]:


y_pred[60]


# In[111]:


cls_true[60]


# In[ ]:




