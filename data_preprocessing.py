#%%
import pandas as pd 
import zipfile
from nltk.corpus import stopwords
import re
import numpy as np 
from sklearn.metrics.classification import  log_loss
from sklearn.metrics.cluster import normalized_mutual_info_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

#%%
zf = zipfile.ZipFile("training_variants.zip")

#%%
for name in zf. namelist():
    print(name)

#%%
df = pd.read_csv(zf.open('training_variants'))

#%%
df.head()

#%%
def missing_data_function(frame):
        total = frame.isnull().sum().sort_values(ascending=False)
        percent = (frame.isnull().sum()*100 / frame.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data

#%%
print(missing_data_function(df))

#%%
df.shape

#%%
text_zf=zipfile.ZipFile("training_text.zip")

#%%
for name in text_zf.namelist():
    print(name)


#%%
df_text=pd.read_csv("training_text.txt" ,sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)


#%%
df_text.head()

#%%
missing_data_function(df_text)

#%%
df_text.keys()

#%%
Stopwords = set(stopwords.words('english'))

#%%

print(Stopwords)
#%%

def text_preprocessing(text):
    if type(text) is str:
       
        d={'[^a-zA-Z0-9\n]':" ",
            '["\',\.<>()=*#^:;%µ?|&!-123456789]€$¼�âœ“':""}
        for j in d.keys():
                text=re.sub(j,d[j],text)
        text= text.lower()
        t=""
        for word in text.split():
            if not word in Stopwords:
                t += word + " "
    else : 
        t="" 
    return t 

#%%
df_text['clean_text']=df_text['TEXT'].apply(text_preprocessing)

#%%
text_preprocessing(df_text['TEXT'][0])

#%%
df_text.head()

#%%
data=pd.merge(df, df_text,on='ID', how='left')

#%%
missing_data_function(data)

#%%
data=data.dropna()

#%%
data['Class'].value_counts().plot(kind='bar')

#%%
data.head(n=20)

#%%
data=data.drop(columns=["TEXT"])

#%%
# bench mark 
shape=data.shape[0]
unique=data["Class"].unique()
random_uniform=np.random.choice(unique,shape)


#%%
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html



#%%
normalized_mutual_info_score(data["Class"].values,random_uniform)

#%%
random_pro=np.random.choice(unique,shape,p=[ data["Class"].value_counts()[i]*(1/shape) for i in unique])


#%%
normalized_mutual_info_score(data["Class"].values,random_pro)


#%%

cm=confusion_matrix(data["Class"].values,random_pro)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#%%
labels = [1,2,3,4,5,6,7,8,9]
sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

#%%
cm

#%%
