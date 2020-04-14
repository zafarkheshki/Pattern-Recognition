import csv
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

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

kf = KFold(n_splits=10)
a = 0
c_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for train_index, test_index in kf.split(data):
    train_data, test_data = data[train_index], data[test_index]   
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_1 = clf.predict(test_data[:,1:])
    loo_mat = confusion_matrix(test_data[:,0], predicted_class_1, labels = [0,1,2])

    c_mat +=  loo_mat
   
    acc = 0
    for i in range(0,test_data.shape[0]):
            if (test_data[i,0] == predicted_class_1[i]):
                acc = acc + 1
    a +=1          
    print('Accuracy for fold %s : %f' % (a, acc/float(test_data.shape[0])))
print('Confusion Matrix')
print(c_mat)
accuracy = (c_mat[0,0] + c_mat[1,1] + c_mat[2,2]) / 150
print('General Accuracy: %f' % (accuracy))

kf = KFold(n_splits=10)
a = 0
c_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for train_index, test_index in kf.split(data):
    train_data, test_data = data[train_index], data[test_index]   
    clf = svm.SVC(gamma='scale')
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_2 = clf.predict(test_data[:,1:])
    loo_mat = confusion_matrix(test_data[:,0], predicted_class_2, labels = [0,1,2])

    c_mat +=  loo_mat
   
    acc = 0
    for i in range(0,test_data.shape[0]):
            if (test_data[i,0] == predicted_class_2[i]):
                acc = acc + 1
    a +=1          
print('Accuracy for fold %s : %f' % (a, acc/float(test_data.shape[0])))
print('Confusion Matrix')
print(c_mat)
accuracy = (c_mat[0,0] + c_mat[1,1] + c_mat[2,2]) / 150
print("General Accuracy: ",(accuracy))

kf = KFold(n_splits=10)
a = 0
# From scipy.sparse import csr_matrix
c_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for train_index, test_index in kf.split(data):
    # Print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]   
    # Print(train_data, test_data)
    # clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_3 = clf.predict(test_data[:,1:])
    # Print(predicted_class)
    loo_mat = confusion_matrix(test_data[:,0], predicted_class_3, labels = [0,1,2])

    c_mat +=  loo_mat
   
    acc = 0
    for i in range(0,test_data.shape[0]):
            if (test_data[i,0] == predicted_class_3[i]):
                acc = acc + 1
    a +=1          
print('Accuracy for fold %s : %f' % (a, acc/float(test_data.shape[0])))
print('Confusion Matrix')
print(c_mat)
accuracy = (c_mat[0,0] + c_mat[1,1] + c_mat[2,2]) / 150
print('General Accuracy: %f' % (accuracy))

# AdaBoost
kf = KFold(n_splits=10)
a = 0
# From scipy.sparse import csr_matrix
c_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# csr_mat = csr_matrix((3,3), dtype=np.int8)
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]   
    #print(train_data, test_data)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    #clf = AdaBoostClassifier()
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_4 = clf.predict(test_data[:,1:])
    #print(predicted_class)
    loo_mat = confusion_matrix(test_data[:,0], predicted_class_4, labels = [0,1,2])

    c_mat +=  loo_mat
   
    acc = 0
    for i in range(0,test_data.shape[0]):
            if (test_data[i,0] == predicted_class_4[i]):
                acc = acc + 1
    a +=1          
    print('Accuracy for fold %s : %f' % (a, acc/float(test_data.shape[0])))
print('Confusion Matrix')
print(c_mat)
accuracy = (c_mat[0,0] + c_mat[1,1] + c_mat[2,2]) / 150
print('General Accuracy: %f' % (accuracy))
kf = KFold(n_splits=10)
a = 0
c_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for train_index, test_index in kf.split(data):
    train_data, test_data = data[train_index], data[test_index]   
    clf = GaussianNB()
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_5 = clf.predict(test_data[:,1:])
    loo_mat = confusion_matrix(test_data[:,0], predicted_class_5, labels = [0,1,2])

    c_mat +=  loo_mat
   
    acc = 0
    for i in range(0,test_data.shape[0]):
            if (test_data[i,0] == predicted_class_5[i]):
                acc = acc + 1
    a +=1          
print('Accuracy for fold %s : %f' % (a, acc/float(test_data.shape[0])))
print('Confusion Matrix')
print(c_mat)
accuracy = (c_mat[0,0] + c_mat[1,1] + c_mat[2,2]) / 150
print('General Accuracy: %f' % (accuracy) )

#################Majority voting####################
c_mat_acu_ALL =  np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
for train_index, test_index in kf.split(data):
    train_data, test_data = data[train_index], data[test_index]   

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_1 = clf.predict(test_data[:,1:])
    
    clf = svm.SVC(gamma='scale')
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_2 = clf.predict(test_data[:,1:])
    
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_3 = clf.predict(test_data[:,1:])
    
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_4 = clf.predict(test_data[:,1:])
    
    clf = GaussianNB() 
    clf = clf.fit(train_data[:,1:], train_data[:,0])
    predicted_class_5 = clf.predict(test_data[:,1:])
    

    pc_all = np.zeros((1,15))    
    for j in range (0,len(predicted_class_1)):  
        vote_bin = np.zeros((1,3))
        vote_bin[0,int(predicted_class_1[j])] = vote_bin[0,int(predicted_class_1[j])] + 1
        vote_bin[0,int(predicted_class_2[j])] = vote_bin[0,int(predicted_class_2[j])] + 1   
        vote_bin[0,int(predicted_class_3[j])] = vote_bin[0,int(predicted_class_3[j])] + 1
        vote_bin[0,int(predicted_class_4[j])] = vote_bin[0,int(predicted_class_4[j])] + 1
        vote_bin[0,int(predicted_class_5[j])] = vote_bin[0,int(predicted_class_5[j])] + 1
        pc_all[0,j] = np.argmax(vote_bin)
        
    print(pc_all)
    c_mat = confusion_matrix(test_data[:,0], pc_all[0,:], labels=[0,1,2])
    c_mat_acu_ALL += c_mat    
print('Confusion Matrix')
print(c_mat_acu_ALL)
accuracy = (c_mat_acu_ALL[0,0] + c_mat_acu_ALL[1,1] + c_mat_acu_ALL[2,2]) / 150
print('General Accuracy: %f' % (accuracy))
