import csv
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneOut

csv.register_dialect('myDialect',
delimiter = '\t',
skipinitialspace=True)

# Read data
data_table = []
with open('Fisher.txt', 'r') as csvFile:
    reader = csv.reader(csvFile, dialect='myDialect')
    data_table = list(reader)
csvFile.close()

header = data_table[0]
del data_table[0]

data = np.zeros((len(data_table), len(data_table[0])))

for i in range(0,len(data_table)):
    tmp = data_table[i]
    for j in range(0,len(tmp)):
        data[i,j] = float(tmp[j])

feature=data[:,[1,2,3,4]]
labels=data[:,[0]]


# Perform Leave One Out Validation on just the Decision Tree Classifier. 
LOO=LeaveOneOut()
number_of_iterations=LOO.get_n_splits(feature)
total_score=0;
d3=tree.DecisionTreeClassifier()
for train_index,test_index in LOO.split(feature):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_features, test_features = feature[train_index], feature[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    d3=tree.DecisionTreeClassifier()
    clf=d3.fit(train_features,train_labels)
    total_score+=clf.score(test_features,test_labels)
  
score = mean_score=(total_score/number_of_iterations)
print("D3 + leave one cross validation:",score)


# Perform Cross Validation with 10 folds on the Decision Tree Classifier
from sklearn.model_selection import cross_val_score
d3=tree.DecisionTreeClassifier()
scores = cross_val_score(d3, feature, labels, cv=10)
scores = scores.mean()
print("D3+ fold 10 cross_validation:",scores)


# Perform Leave One Out validation on PCA-Decision Tree classifier. 
np.seterr(divide='ignore', invalid='ignore')


total_score=0
for train_index,test_index in LOO.split(feature):
    train_features, test_features = feature[train_index], feature[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    
    pca = PCA(n_components=1)
    pca_train_set= pca.fit_transform(train_features) 
    pca_test_set=  pca.fit_transform(test_features)
    
    clf = d3.fit(pca_train_set,train_labels)
    prediction_pca=clf.predict(pca_test_set)
    total_score+=accuracy_score(test_labels,prediction_pca) 
score = mean_score=(total_score/number_of_iterations)
print("PCA Scores + leave one cross_validation:",score)


# Cross Validation with 10 folds on the PCA-Decision Tree Classifier

pca = PCA()
pca_features= pca.fit_transform(feature) 

scores = cross_val_score(d3, pca_features, labels, cv=10)
scores = scores.mean()
print("PCA Scores + 10 fold cross_validation:",scores)

# Perform Leave One Out validation for the LDA - Decision Tree Classifier

total_score=0
for train_index,test_index in LOO.split(feature):
    train_features, test_features = feature[train_index], feature[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    lda  = LDA()
    lda=lda.fit(train_features,train_labels.ravel())
    lda_train_set = lda.transform(train_features)
    lda_test_set = lda.transform(test_features)
    
    clf_lda=d3.fit(lda_train_set,train_labels)
    prediction_lda=clf_lda.predict(lda_test_set)
    total_score+=accuracy_score(test_labels,prediction_lda) 
mean_score=(total_score/number_of_iterations)
score = mean_score
print("LDA Scores + leave one cross_validation:",score)


# Perform Cross Validation for 10 folds for the LDA-Decision Tree Classifier

lda = LDA()
lda_features=lda.fit_transform(feature,labels.ravel()) 

scores = cross_val_score(d3, lda_features, labels, cv=10)
scores = scores.mean()
print("LDA Scores + 10 fold cross_validation:",scores)


