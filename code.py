#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 17:50:16 2018

@author: divyansha
"""

import nltk
import pandas as pd
import numpy as np
import string
import re
import nltk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

df = pd.read_csv('testdata.manual.2009.06.14.csv', header=None,parse_dates=[2])
df_cols=['polarity', 'id','date','query','user','text']
df.columns = df_cols
df.drop(['polarity', 'id','date','query','user'], axis=1, inplace=True)
print(done)
#remove  ‘@’
import re
for i in df.index:
    sub = re.sub(r'@[A-Za-z0-9_]+','',df.text[i])
    df["text"][i]=sub
# remove '#' and links
for i in df.index:
    sub = re.sub('https?://[A-Za-z0-9./]+','',df.text[i])
    df["text"][i]=sub
for i in df.index:
    sub = re.sub("[^a-zA-Z]", " ", df.text[i])
    df["text"][i]=sub   

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer#specially tokenize twitter tweets
tt=TweetTokenizer()
df['text']=df['text'].apply(tt.tokenize)
df['text-filtered'] = ""

#remove punctuations
for w in df.index:
    rem=[i for i in df['text-filtered'][w] if i not in punctuations]
    df['text-filtered'][w]=rem
    
#remove stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
for i in df.index:
    filtered_sentence = [w for w in df['text'][i] if not w in stop_words] 
    df['text-filtered'][i] = filtered_sentence
    
#steamming of data
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stemmer = PorterStemmer()
df['text-stem']=df['text-filtered'].apply(lambda x : [porter_stemmer.stem(y) for y in x])

#dividing into n-grams (4)
four=[]
for j in range(498):
    fourgrams=nltk.collocations.QuadgramCollocationFinder.from_words(df['text-filtered'][j])
    df['four']=""
    for fourgram,value in fourgrams.ngram_fd.items():
        four.append(list(fourgram))
        
df1 = pd.DataFrame(four) 
df1.columns=['1','2','3','4']

df1['New']=df1['1']+" "+df1['2']+" "+df1['3']

#This x will be vectorized using tf-idf
X_word_vector = df1.iloc[:,4].values
y = df1.iloc[:, 3].values

TfidfVect = TfidfVectorizer()
X_word_vector = TfidfVect.fit_transform(X_word_vector)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labels = labelencoder_y.fit_transform(y)

x_train_word_vector, x_test_word_vector, y_train, y_test = train_test_split(X_word_vector, labels, test_size=0.2, random_state=4)

from sklearn.metrics import confusion_matrix 
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 4,criterion='gini').fit(x_train_word_vector, y_train)
dtree_predictions = dtree_model.predict(x_test_word_vector)
 
# creating a confusion matrix
cm_dtree_model = confusion_matrix(y_test, dtree_predictions)

accuracy_dtree_model = dtree_model.score(x_test_word_vector, y_test)

label=[]
accuracy=[]
label.append('dtree_model')
accuracy.append(accuracy_dtree_model)

 # training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train_word_vector, y_train)
svm_predictions = svm_model_linear.predict(x_test_word_vector)
 
# model accuracy for X_test  
accuracy_svm = svm_model_linear.score(x_test_word_vector, y_test)
 
# creating a confusion matrix
cm_svm = confusion_matrix(y_test, svm_predictions)

label.append('svm_model_linear')
accuracy.append(accuracy_svm)

#pickling of our model 
import pickle
save_classifier = open("svm_model_linear.pickle","wb")
pickle.dump(svm_model_linear, save_classifier)
save_classifier.close()

from sklearn import metrics
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
k_range = list(range(1, 100))
scores = []
knn = KNeighborsClassifier(n_neighbors = 50).fit(x_train_word_vector, y_train)
 
# accuracy on X_test
accuracy_knn = knn.score(x_test_word_vector, y_test)
print (accuracy_knn)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_word_vector, y_train)
    y_pred = knn.predict(x_test_word_vector)
    scores.append(metrics.accuracy_score(y_test, y_pred))
 
# creating a confusion matrix
knn_predictions = knn.predict(x_test_word_vector) 
cm_knn = confusion_matrix(y_test, knn_predictions)

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt


# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.savefig("knn_accuracy.png")

label.append('knn')
accuracy.append(accuracy_knn)

#pickling of our model 
import pickle
save_classifier = open("knn.pickle","wb")
pickle.dump(knn, save_classifier)
save_classifier.close()

x_train_word_vector=x_train_word_vector.toarray()
x_test_word_vector=x_test_word_vector.toarray()
# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(x_train_word_vector, y_train)
gnb_predictions = gnb.predict(x_test_word_vector)
 
# accuracy on X_test
accuracy_gnb = gnb.score(x_test_word_vector, y_test)
print (accuracy_gnb)
 
# creating a confusion matrix
cm_gnb = confusion_matrix(y_test, gnb_predictions)

#pickling of our model 
import pickle
save_classifier = open("gnb.pickle","wb")
pickle.dump(gnb, save_classifier)
save_classifier.close()

label.append('gnb')
accuracy.append(accuracy_gnb)

from sklearn import svm
svc = svm.SVC(decision_function_shape='ovo')
svc.fit(x_train_word_vector, y_train) 
accuracy_svc = svc.score(x_test_word_vector, y_test)


#pickling of our model 
import pickle
save_classifier = open("svc.pickle","wb")
pickle.dump(svc, save_classifier)
save_classifier.close()

label.append('svc')
accuracy.append(accuracy_svc)

lin_svc = svm.LinearSVC()
lin_svc.fit(x_train_word_vector, y_train) 
accuracy_lin_svc = lin_svc.score(x_test_word_vector, y_test)

#pickling of our model 
import pickle
save_classifier = open("lin_svc.pickle","wb")
pickle.dump(lin_svc, save_classifier)
save_classifier.close()

label.append('lin_svc')
accuracy.append(accuracy_lin_svc)

lin_clf_cr = svm.LinearSVC(multi_class='crammer_singer')
lin_clf_cr.fit(x_train_word_vector, y_train) 
accuracy_lin_clf_cr = lin_clf_cr.score(x_test_word_vector, y_test)


#pickling of our model 
import pickle
save_classifier = open("lin_clf_cr.pickle","wb")
pickle.dump(lin_clf_cr, save_classifier)
save_classifier.close()

label.append('lin_clf_cr')
accuracy.append(accuracy_lin_clf_cr)

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(x_train_word_vector, y_train) 
accuracy_rbf_svc = rbf_svc.score(x_test_word_vector, y_test)

#pickling of our model 
import pickle
save_classifier = open("rbf_svc.pickle","wb")
pickle.dump(rbf_svc, save_classifier)
save_classifier.close()

label.append('rbf_svc')
accuracy.append(accuracy_rbf_svc)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(x_train_word_vector, y_train)
print(rf.feature_importances_)
accuracy_rf = rf.score(x_test_word_vector, y_test)
print (accuracy_rf)

label.append('rf')
accuracy.append(accuracy_rf)


#pickling of our model 
import pickle
save_classifier = open("rf.pickle","wb")
pickle.dump(rf, save_classifier)
save_classifier.close()


X = df1.iloc[:,:3].values
y = df1.iloc[:, 3].values

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labels = labelencoder_y.fit_transform(y)
X=pd.DataFrame(X)
X=X.apply(LabelEncoder().fit_transform)


x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=4)

from sklearn.metrics import confusion_matrix 
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier 
dtree_model2 = DecisionTreeClassifier(max_depth = 2).fit(x_train, y_train)
dtree_predictions2 = dtree_model2.predict(x_test)
 
# creating a confusion matrix
cm_dtree_model2 = confusion_matrix(y_test, dtree_predictions)

accuracy_dtree_model2 = dtree_model2.score(x_test, y_test)

label.append('dtree_model2')
accuracy.append(accuracy_dtree_model2)

rf2 = RandomForestClassifier(max_depth=20, random_state=0)
rf2.fit(x_train, y_train)
print(rf2.feature_importances_)
accuracy_rf2 = rf2.score(x_test, y_test)

#pickling of our model 
import pickle
save_classifier = open("rf2.pickle","wb")
pickle.dump(rf2, save_classifier)
save_classifier.close()

label.append('rf2')
accuracy.append(accuracy_rf2)

from sklearn import metrics
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
k_range = list(range(1,100))
scores2 = []
knn2 = KNeighborsClassifier(n_neighbors = 50).fit(x_train, y_train)
 
# accuracy on X_test
accuracy_knn2 = knn2.score(x_test, y_test)
print (accuracy_knn2)
for k in k_range:
    knn2 = KNeighborsClassifier(n_neighbors=k)
    knn2.fit(x_train, y_train)
    y_pred = knn2.predict(x_test)
    scores2.append(metrics.accuracy_score(y_test, y_pred))
 
# creating a confusion matrix
knn_predictions2 = knn2.predict(x_test) 
cm_knn2 = confusion_matrix(y_test, knn_predictions2)

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt


# plot the relationship between K and testing accuracy
plt.plot(k_range, scores2)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.savefig("knn2_accuracy.png")

label.append('knn2')
accuracy.append(accuracy_knn2)

#pickling of our model 
import pickle
save_classifier = open("knn2.pickle","wb")
pickle.dump(knn2, save_classifier)
save_classifier.close()

import matplotlib.pyplot as plt

index = np.arange(len(label))
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, accuracy)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=90)
    plt.title('Accuracy of different models')
    plt.savefig("model_accuracy.png")
    plt.show()
    
plot_bar_x()







