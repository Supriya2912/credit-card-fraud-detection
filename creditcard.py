#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pip install --upgrade bottleneck


# In[5]:


df = pd.read_csv('creditcard_2023.csv')


# In[6]:


df.head()


# In[5]:


df.info()


# In[7]:


df.isnull().sum() # to check missing values 


# In[8]:


x = df.drop(['id','Class'],axis=1,errors='ignore')
y= df['Class']


# In[9]:


print(x.columns.tolist())


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42)


# In[11]:


x_train.shape


# In[12]:


x_test.shape


# In[13]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[14]:


print(pd.Series(y_train).value_counts(normalize=True))


# In[15]:


rf_model = RandomForestClassifier(
n_estimators = 100,
max_depth = 10,
min_samples_split = 5,
random_state=42
)


# In[17]:


cv_scores = cross_val_score(rf_model,x_train_scaled,y_train,cv = 5,scoring = 'f1')
print("\nCross-validation F1 scores:",cv_scores)
print("Average F1 score:",np.mean(cv_scores))


# In[18]:


rf_model.fit(x_train_scaled,y_train)


# In[ ]:





# In[19]:


y_pred = rf_model.predict(x_test_scaled)


# In[20]:


print(classification_report(y_test,y_pred))


# In[21]:


plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[22]:


importance = rf_model.feature_importances_
feature_imp = pd.DataFrame({
    'Feature': x.columns,
    'Importance': importance
}).sort_values('Importance',ascending=False)


# In[23]:


feature_imp.head()


# In[24]:


plt.figure(figsize=(10,6))
sns.barplot(data=feature_imp,x = 'Importance',y='Feature')
plt.title('Feature Importance Ranking')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()


# In[26]:



plt.figure(figsize=(12, 8))
correlation_matrix = x.corr()  # Fixed assignment operator
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f")  # Fixed typos in sns and cmap
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()


# In[27]:


y_pred_proba= rf_model.predict_proba(x_test_scaled) [:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc= auc(fpr, tpr)


# In[29]:



plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')  # Fixed lw, commas, and label
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle='--')  # Fixed lw and added comma
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])  # Fixed ylim (was 10.0 instead of 0.0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




