#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries as needed

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, roc_auc_score, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from src.feature_importance_plot import feature_importance_plot
from src.learning_curve_plot import learning_curve_plot


# # Read and explore de data

# In[2]:


#Read data file

audio = pd.read_csv("../data/audio_data_w_genres.csv")
audio.head()


# In[3]:


#Drop Filename, unnecessary 

audio = audio.drop(labels='filename',axis=1)
audio.head()


# In[4]:


audio.info()


# In[5]:


#Boxplot with the beats
sns.boxplot(x= 'label', y='beats', data=audio)


# In[6]:


#Boxplot with the tempo
sns.boxplot(x= 'label', y='tempo', data=audio)


# In[7]:


#Boxplot with the spectral_bandwidth
sns.boxplot(x= 'label', y='spectral_bandwidth', data=audio)


# In[8]:


#Encode the label column!

label_encoder = preprocessing.LabelEncoder()
audio['label_encoded'] = label_encoder.fit_transform(audio['label'])
audio = audio.drop(labels='label',axis=1)
audio.head()


# In[9]:


#Check if there is duplicated data
print("# of duplicated rows of data:", audio[audio.duplicated(keep = False)].shape[0])


# In[10]:


#We have to remove that data
audio = audio[~audio.duplicated(keep = 'first')]


# In[11]:


#Confirm there is no more duplicated data
print("# of duplicated rows of data:", audio[audio.duplicated(keep = False)].shape[0])


# In[12]:


#check if correlated variables
threshold = 0.85 # define threshold

corr_matrix = audio.corr().abs() # calculate the correlation matrix with 
high_corr_var = np.where(corr_matrix >= threshold) # identify variables that have correlations above defined threshold
high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y], round(corr_matrix.iloc[x, y], 2))
                         for x, y in zip(*high_corr_var) if x != y and x < y] # identify pairs of highly correlated variables


high_corr_var


# In[13]:


#Lets drop correlated values
audio = audio.drop(labels='beats',axis=1)
audio = audio.drop(labels='rolloff',axis=1)
audio = audio.drop(labels='spectral_centroid',axis=1)
audio = audio.drop(labels='mfcc2',axis=1)


# In[14]:


#check if correlated variables
threshold = 0.85 # define threshold

corr_matrix = audio.corr().abs() # calculate the correlation matrix with 
high_corr_var = np.where(corr_matrix >= threshold) # identify variables that have correlations above defined threshold
high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y], round(corr_matrix.iloc[x, y], 2))
                         for x, y in zip(*high_corr_var) if x != y and x < y] # identify pairs of highly correlated variables


high_corr_var


# In[15]:


#Shape of data
audio.shape

#987 rows & 24 columns


# # Missing data

# In[16]:


#Do we have missing data? (apparently no) Handle missing data? (no missing data)
audio.isnull().sum()


# In[17]:


#Data information

audio.info()

#3 more audio files mmm, let's ignore them


# In[18]:


#Lets try describing this

audio.describe()


# In[19]:


#Histograms! always helpful

plt.rcParams["figure.figsize"] = (20,20)
audio.hist()

#We're gonna put special attention to mfcc1, something sketchy going on in there


# In[20]:


#Lets check the standard deviation!

audio.std()


# In[21]:


#We identify 2 features with very little std, so let's drop them, for the sake of simplicity

audio = audio.drop(labels='chroma_stft',axis=1)
audio = audio.drop(labels='zero_crossing_rate',axis=1)
audio.std()

#chroma_stft and zero_crossing_rate, you are the weakest link. Good bye.


# In[22]:


#As stated in the project definition, even distribution of the songs (pretty useless to do this I guess?)

audio.label_encoded.value_counts(dropna = True).plot(kind = 'bar')
plt.title("Distribution of the audio genre");
audio.label_encoded.describe()


# In[23]:


#We identified "label" as our target, Yei!


# # Split the data into train and test

# In[24]:


data = audio.drop(["label_encoded"], axis = 1)
target = audio["label_encoded"]

X = data
y = target


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)


# In[26]:


target


# In[27]:


data


# In[28]:


# print the shape of the training data

print("Training Data")
print("Shape of X_train", X_train.shape)
print("Shape of y_train", y_train.shape)


# In[29]:


# print the shape of the test data 

print("Test Data")
print("Shape of X_test", X_test.shape)
print("Shape of y_test", y_test.shape)


# # Model Training and Performance Evaluation

# Decision Tree Classifier

# In[30]:


#Lets try a decision tree classifier! YEEEEAAA
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


# In[31]:


# Print the first five predicted vs actual values
print("first five predicted values:", y_pred_dt[0:5])
print("first five actual values:", list(y_test[0:5]))


# In[32]:


#Check evaluation matrics test 
print("accuracy:", round(accuracy_score(y_test, y_pred_dt), 2))
print("recall:", round(recall_score(y_test, y_pred_dt, average = 'macro'), 2))
print("precision:", round(precision_score(y_test, y_pred_dt, average = 'macro'), 2))
print("f1-score:", round(f1_score(y_test,y_pred_dt, average = 'macro'), 2))


# In[33]:


#Confusion matrix
plot_confusion_matrix(dt, 
                      X_test, 
                      y_test,
                      cmap = plt.cm.Blues);


# In[34]:


#Plot learning curve
learning_curve_plot(dt, X_train, y_train,None)


# XGBoost Classifier

# In[35]:


#Now let's try with XGBoot classifier
xgb_model = XGBClassifier(max_depth = 5, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)


# In[36]:


# Print the first five predicted vs actual values
print("first five predicted values:", xgb_predictions[0:5])
print("first five actual values:", list(y_test[0:5]))


# In[37]:


#Check evaluation matrics
print("accuracy:", round(accuracy_score(y_test, xgb_predictions), 2))
print("recall:", round(recall_score(y_test, xgb_predictions, average = 'macro'), 2))
print("precision:", round(precision_score(y_test, xgb_predictions, average = 'macro'), 2))
print("f1-score:", round(f1_score(y_test,xgb_predictions, average = 'macro'), 2))


# In[38]:


#Confussion matrix
plot_confusion_matrix(xgb_model, 
                      X_test, 
                      y_test,
                      cmap = plt.cm.Blues);


# In[39]:


#Plot learning curve
learning_curve_plot(xgb_model, X_train, y_train,None)


# In[40]:


#It is improving, but still not what we want


# Random Forest Classifier

# In[41]:


#Let's try with random forest
rforest_model = RandomForestClassifier().fit(X_train, y_train)
rforest_predictions = rforest_model.predict(X_test)


# In[42]:


#Check evaluation matrix test
print("accuracy:", round(accuracy_score(y_test, rforest_predictions), 2))
print("recall:", round(recall_score(y_test, rforest_predictions, average = 'macro'), 2))
print("precision:", round(precision_score(y_test, rforest_predictions, average = 'macro'), 2))
print("f1-score:", round(f1_score(y_test, rforest_predictions, average = 'macro'), 2))


# In[43]:


#Confusion matrix 
plot_confusion_matrix(rforest_model, 
                      X_test, 
                      y_test,
                      cmap = plt.cm.Blues);


# In[44]:


#Check with genres are good to be predicted by our model 
print(classification_report(rforest_predictions, y_test))
label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# In[45]:


#Plot learning curve
learning_curve_plot(rforest_model, X_train, y_train,None)


# # Feature importance plot

# In[46]:


#Decision Tree
feature_importance_plot(dt, X_train, 10)


# In[47]:


#XGBoost
feature_importance_plot(xgb_model, X_train, 10)


# In[48]:


#Random Forest
feature_importance_plot(rforest_model, X_train, 10)


# # Hyperparameter Tuning

# In[49]:


#Let's search for the best parametes
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=0),
                           {
                              'n_estimators':np.arange(5,100,5),
                              'max_features':np.arange(0.1,1.0,0.05),
                            },cv=5, scoring="r2",verbose=1,n_jobs=-1, 
                             n_iter=50, random_state = 0
                           )
random_search.fit(X_train,y_train)


# In[50]:


#Let's search for the best parametes to tune
random_search.best_params_


# In[51]:


#What will be the best score
random_search.best_score_


# # Random Forest Classifier. Second Iteration

# In[52]:


#Let's try random forest with the new parametes
rforest_model = RandomForestClassifier(**random_search.best_params_).fit(X_train, y_train)
rforest_predictions = rforest_model.predict(X_test)


# In[53]:


# Print the first five predicted vs actual values
print("first five predicted values:", rforest_predictions[0:5])
print("first five actual values:", list(y_test[0:5]))


# In[54]:


#Check evaluation matrix test
print("accuracy:", round(accuracy_score(y_test, rforest_predictions), 2))
print("recall:", round(recall_score(y_test, rforest_predictions, average = 'macro'), 2))
print("precision:", round(precision_score(y_test, rforest_predictions, average = 'macro'), 2))
print("f1-score:", round(f1_score(y_test, rforest_predictions, average = 'macro'), 2))


# In[56]:


#We check the confusion matrix again
plot_confusion_matrix(rforest_model, 
                      X_test, 
                      y_test,
                      cmap = plt.cm.Blues);


# In[57]:


#Check with genres are good to be predicted by our model 
print(classification_report(rforest_predictions, y_test))
label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# In[58]:


#Plot the learning curve again
learning_curve_plot(rforest_model, X_train, y_train,None)


# In[ ]:




