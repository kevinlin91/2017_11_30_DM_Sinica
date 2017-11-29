
# coding: utf-8

# <h1>Supervised Learning (Classification)</h1>
# <ul>
#     <li>Decision Tree</li>
#     <li>Naive Bayes</li>
#     <li>Random Forest</li>
#     <li>Support Vector Machine</li>
# </ul>

# <h1> Datasets </h1>
# <ul>
#     <li>Iris(classification)</li>
# </ul>

# In[1]:


from sklearn.datasets import load_iris
iris_data = load_iris()


# <h2>Iris</h2>

# In[2]:


print iris_data.DESCR


# In[3]:


print iris_data.target
print iris_data.target_names
print iris_data.data
print iris_data.feature_names


# <h2>Some problems before applying algorithms</h2>
# <ul>
#     <li>Feature Engineering</li>
#     <li>Cross Validation</li>
#     <li>Imbalanced Data</li>
# </ul>

# <h2>Cross Validation</h2>
# <li>Leave One(p) Out Cross Validation (LOOCV)</li>
# <li>k-fold Cross Validation</li>
# 
# <p style='color:red'>sklearn.model_selection</p>

# Leave One Out Cross Validation

# In[4]:


from sklearn.model_selection import LeaveOneOut
#in sklearn most of time we use x to present training data
train_data = load_iris().data
#in sklearn most of time we use y to present label
labels = load_iris().target
loocv = LeaveOneOut()
print loocv.get_n_splits(train_data)


# In[5]:


for train_index, test_index in loocv.split(train_data):
    print train_index, test_index


# In[6]:


for train_index, test_index in loocv.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]
    print len(X_train), len(X_test), len(Y_train), len(Y_test)


# <h2>Cross Validation</h2>

# In[7]:


from sklearn.model_selection import KFold

five_fold = KFold(n_splits=5)
print five_fold.get_n_splits(train_data)


# In[8]:


for train_index, test_index in five_fold.split(train_data):
    print train_index, test_index


# In[9]:


for train_index, test_index in five_fold.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]
    print len(X_train), len(X_test), len(Y_train), len(Y_test)


# <h2>Imbalanced Data</h2>
# <li>Over Sampling</li>
# <li>Under Sampling</li>
# <li style="color:red">Synthetic Minority Over-sampling Technique (SMOTE) </li>

# In[10]:


import numpy as np
from imblearn.over_sampling import SMOTE
print np.count_nonzero(labels == 0)
print np.count_nonzero(labels == 1)
print np.count_nonzero(labels == 2)


# In[11]:


new_train_data = load_iris().data[:60]
new_labels = load_iris().target[:60]
print new_train_data
print new_labels


# In[12]:


sm = SMOTE(k_neighbors=5)
train_sample, labels_sample = sm.fit_sample(new_train_data,new_labels)
print new_labels
print labels_sample
print np.count_nonzero(labels_sample==1)


# <h2>Decision Tree</h2>
# <ul>
#     <li>sklearn.tree.DecisionTreeClassifier</li>
#     <li>input
#     <ul>
#         <li>Training Data</li>
#         <li>Label</li>
#     </ul>
# </ul>

# criterion<br>
# &nbsp;&nbsp;&nbsp;gini, entropy<br>
# 

# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


decision_clf = DecisionTreeClassifier()
decision_clf.fit(iris_data.data,iris_data.target)


# In[15]:


from sklearn.model_selection import cross_val_score
decision_clf = DecisionTreeClassifier()
decision_score = cross_val_score(decision_clf, iris_data.data, iris_data.target, cv=5)
print decision_score


# In[16]:


clf = DecisionTreeClassifier()
clf.fit(iris_data.data,iris_data.target)
print clf.predict(iris_data.data[:100])
print clf.predict_proba(iris_data.data[:50])


# <h2>Naive Bayes</h2>
# <ul>
#     <li>sklearn.naive_bayes
#     <ul>
#         <li>Gaussian</li>
#         <li>Multinomial</li>
#         <li>Bernoulli</li>
#     </ul>
#     <li>input
#     <ul>
#         <li>Training Data</li>
#         <li>Label</li>
#     </ul>
# </ul>

# In[17]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(iris_data.data, iris_data.target)


# In[18]:


print nb_clf.predict(iris_data.data[:50])
print nb_clf.predict_proba(iris_data.data[:50])


# In[19]:


nb_clf = GaussianNB()
GaussianNB_score = cross_val_score(nb_clf, iris_data.data, iris_data.target, cv=5)
print GaussianNB_score


# <h2>Random Forest (Ensemble)</h2>
#     <li>sklearn.ensemble</li>
#     <li>input
#     <ul>
#         <li>Training Data</li>
#         <li>Label</li>
#     </ul>

# n_estimators<br>
# &nbsp;&nbsp;&nbsp;The number of trees in the forest<br>
# criterion<br>
# &nbsp;&nbsp;&nbsp;gini, entropy<br>
# max_features<br>
# 

# In[20]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(iris_data.data,iris_data.target)


# In[21]:


print rf_clf.predict(iris_data.data[:50])
print rf_clf.predict_proba(iris_data.data[:50])


# In[22]:


rf_clf = RandomForestClassifier()
rf_score = cross_val_score(rf_clf, iris_data.data, iris_data.target, cv=5)
print rf_score


# <h2>Support Vector Machine</h2>
#     <li>sklearn.svm</li>
#     <li>input
#     <ul>
#         <li>Training Data</li>
#         <li>Label</li>
#     </ul>

# C<br>
# &nbsp;&nbsp;&nbsp;Penalty parameter<br>
# kernel<br>
# &nbsp;&nbsp;&nbsp;linear, polynomial, rbf, sigmoid<br>
# gamma<br>
# &nbsp;&nbsp;&nbsp;Kernel coefficient 

# In[23]:


from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(iris_data.data,iris_data.target)
print svm_clf.predict(iris_data.data[:50])


# In[24]:


svm_clf = SVC(probability=True)
svm_clf.fit(iris_data.data,iris_data.target)
print svm_clf.predict_proba(iris_data.data[:50])


# In[25]:


svm_clf = SVC()
svm_score = cross_val_score(clf, iris_data.data, iris_data.target, cv=5)
print svm_score


# <h2>Kaggle Example: Titanic</h2>
# <p>https://www.kaggle.com/c/titanic</p>
# <ul>
# <li>Training Data (With Label)</li>
# <li>Unseen Data (Without Label)</li>
# <li>Other Informations</li>
# </ul>

# In[26]:


import csv

header = list()
content = list()
with open('train.csv','r') as f:
    
    csv_write = csv.reader(f)
    for line in csv_write:
        content.append(line)
    header = content.pop(0)
    
print header
print content
print len(content)


# In[27]:


training_data = list()
labels = list()

labels = [ row[1] for row in content]

for index,row in enumerate(content):
    training_data.append([])
    #pclass
    training_data[index].append(int(row[2]))
    #gender
    try:
        if row[4] == 'male':
            training_data[index].append(1)
        elif row[4] == 'female':
            training_data[index].append(0)
    except:
        print index,row[4]

print training_data
print labels


# In[28]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#support vector machine
svm_clf = SVC(C=8,gamma=0.125)
svm_score = cross_val_score(svm_clf,training_data,labels,cv=5)

#random forest
rf_clf = RandomForestClassifier()
rf_score = cross_val_score(rf_clf,training_data,labels,cv=5)

#naive bayse
nb_clf = GaussianNB()
nb_score = cross_val_score(nb_clf,training_data,labels,cv=5)

#decision tree
decision_clf = DecisionTreeClassifier()
decision_score = cross_val_score(decision_clf,training_data,labels,cv=5)

print svm_score
print rf_score
print nb_score
print decision_score


# In[29]:


nb_clf.fit(training_data,labels)


# In[30]:


import csv
header_test = list()
content_test = list()
with open('test.csv','r') as f:
    
    csv_write = csv.reader(f)
    for line in csv_write:
        content_test.append(line)
    header_test = content_test.pop(0)
    
print header_test
print content_test
print len(content_test)


# In[31]:


testing_data = list()


for index,row in enumerate(content_test):
    testing_data.append([])
    #pclass
    testing_data[index].append(int(row[1]))
    #gender
    try:
        if row[3] == 'male':
            testing_data[index].append(1)
        elif row[3] == 'female':
            testing_data[index].append(0)
    except:
        print index,row[3]

print testing_data


# In[32]:


predictions = nb_clf.predict(testing_data)
print predictions


# In[33]:


with open('./result.csv','w') as f:
    f.write("PassengerId,Survived\n")
    for index, prediction in enumerate(predictions):
        output_str = str(content_test[index][0]) + "," + str(prediction) + "\n"
        f.write(output_str)

