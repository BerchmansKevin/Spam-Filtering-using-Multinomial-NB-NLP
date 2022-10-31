#!/usr/bin/env python
# coding: utf-8

# # ` Spam Filtering using Multinomial NB`

# ## Step-1

# In[1]:


#import necessary module
import pandas as pd


# In[2]:


df = pd.read_csv("SMSSpamCollection.csv",encoding='latin-1')
df.head()


# In[3]:


df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[4]:


df.head()


# ## Step-2

# In[5]:


#count the sms messages
df['text'].value_counts().sum()


# ## Step-3

# In[6]:


#use groupby()
df.groupby(['label']).count()


# ## Step-4

# In[7]:


y = df['label']


# In[8]:


X = df['text']


# In[9]:


#split the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# ## Step-5

# In[10]:


#function to remove all punctuation and stopwords
from nltk.corpus import stopwords
def process_text(msg):
    punctuations = '''!()-[]:;"\,<>./?@#${}%^_~*&'''
    nopunc = [char for char in msg if char not in punctuations]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split()
        if word.lower() not in stopwords.words('english')]


# In[11]:


import nltk
nltk.download('stopwords')


# ## Step-6

# In[12]:


#create TfidfVectorizer and perform vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

df1 = TfidfVectorizer(use_idf=True,
               analyzer = process_text,
               ngram_range=(1,3),
               min_df=1,
               stop_words = 'english')
df1


# In[13]:


a = df1.fit_transform(X_train)


# In[14]:


a1 = df1.transform(X_test)


# ## Step-7

# In[15]:


#create multinomialNB model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(a,y_train)


# ## Step-8

# In[16]:


#predict labels on test set
y_pred = clf.predict(a1)
y_pred


# ## Step-9

# In[17]:


#find confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[18]:


#find classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ## Step-10

# In[19]:


#modify ngram_range=(1,2) and perform 7 to 9
from sklearn.feature_extraction.text import TfidfVectorizer

df2 = TfidfVectorizer(use_idf=True,
               analyzer = process_text,
               ngram_range=(1,2),
               min_df=1,
               stop_words = 'english')
df2


# In[30]:


b = df2.fit_transform(X_train)
b1= df2.transform(X_test)


# In[31]:


#create multinomialNB model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(b,y_train)


# In[32]:


#predict labels on the test set
y1_pred = clf.predict(b1)
y1_pred


# In[33]:


#print confusion matrix
confusion_matrix(y_test,y1_pred)


# In[34]:


#print classification_report
print(classification_report(y_test,y1_pred))


# In[ ]:




