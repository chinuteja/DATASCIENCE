#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.
get_ipython().run_line_magic('matplotlib', 'inline')
#this line above prepares IPython notebook for working with matplotlib

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns #sets up styles and gives us more plotting options


# In[ ]:





# In[2]:


# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.
get_ipython().run_line_magic('matplotlib', 'inline')
#this line above prepares IPython notebook for working with matplotlib

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns #sets up styles and gives us more plotting options


# In[3]:


df=pd.read_csv("all.csv", header=None,
               names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],
)


# In[4]:


df.head()


# In[5]:


df.dtypes


# In[6]:


df.shape


# In[7]:




df.columns


# In[8]:


type(df.rating), type(df)


# In[9]:


df.rating < 3


# In[10]:


np.sum(df.rating < 3)


# In[11]:


# print 1*True, 1*False


# In[12]:


print(1*True, 1*False)


# In[13]:


np.sum(df.rating < 3)/df.shape[0]


# In[14]:


np.mean(df.rating < 3.0)


# In[15]:


(df.rating < 3).mean()


# In[16]:


df.query("rating > 4.5")


# In[17]:


df[(df.year < 0) & (df.rating > 4)]


# In[18]:


df.dtypes


# In[19]:




df[df.year.isnull()]


# In[20]:




df = df[df.year.notnull()]


# In[21]:


df.shape


# In[22]:




df['rating_count']=df.rating_count.astype(int)
df['review_count']=df.review_count.astype(int)
df['year']=df.year.astype(int


# In[23]:




df['rating_count']=df.rating_count.astype(int)
df['review_count']=df.review_count.astype(int)
df['year']=df.year.astype(int)


# In[24]:


df.dtypes


# In[25]:


# Visualising the data


df.rating.hist();


# In[26]:


sns.set_context("notebook")
meanrat=df.rating.mean()
#you can get means and medians in different ways
print (meanrat, np.mean(df.rating), df.rating.median())
with sns.axes_style("whitegrid"):
    df.rating.hist(bins=30, alpha=0.4);
    plt.axvline(meanrat, 0, 0.75, color='r', label='Mean')
    plt.xlabel("average rating of book")
    plt.ylabel("Counts")
    plt.title("Ratings Histogram")
    plt.legend()
    #sns.despine()


# In[27]:


df.review_count.hist(bins=np.arange(0, 40000, 400))


# In[28]:


df.review_count.hist(bins=np.arange(0, 40000, 400))


# In[29]:


df.review_count.hist(bins=100)
plt.xscale("log")


# In[30]:


plt.scatter(df.year, df.rating, lw=0, alpha=.08)
plt.xlim([1900,2010])
plt.xlabel("Year")
plt.ylabel("Rating"


# In[31]:


plt.scatter(df.year, df.rating, lw=0, alpha=.08)
plt.xlim([1900,2010])
plt.xlabel("Year")
plt.ylabel("Rating")


# In[32]:


alist=[1,2,3,4,5]


# In[33]:


asquaredlist=[i*i for i in alist]
asquaredlist


# In[34]:




plt.scatter(alist, asquaredlist);


# In[35]:


print (type(alist))


# In[36]:


plt.hist(df.rating_count.values, bins=100, alpha=0.5);


# In[37]:


print (type(df.rating_count), type(df.rating_count.values))


# # Vectorization
# ##Numpy arrays are a bit different from regular python lists, and are the bread and butter of data science. Pandas Series are built atop them
# alist + alist
# 

# In[38]:


alist + alist


# In[39]:




np.array(alist)


# In[40]:


np.array(alist)+np.array(alist)


# In[41]:




np.array(alist)**2


# In[42]:


newlist=[]
for item in alist:
    newlist.append(item+item)
newlist


# In[43]:


a=np.array([1,2,3,4,5])
print (type(a))
b=np.array([1,2,3,4,5])

print (a*b)


# In[44]:




a+1



# In[ ]:




