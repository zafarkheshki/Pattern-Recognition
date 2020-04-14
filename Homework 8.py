import csv
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# From sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

csv.register_dialect('myDialect',delimiter = '\t',skipinitialspace=True)

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

# Make train and test data
feature=data[:,[1,2,3,4]]
labels=data[:,[0]]

train_data_features, test_data_features, train_data_labels, test_data_labels = train_test_split(feature, labels, test_size=0.2, random_state=6)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data_features, train_data_labels)
predicted_class = clf.predict(test_data_features)
c_mat = confusion_matrix(test_data_labels, predicted_class, labels = [0,1,2])
print(" Confusion Matrix: D3:")
print(c_mat)
Accuracy_score  = accuracy_score(test_data_labels, predicted_class)
print("Accuracy_score",Accuracy_score)

##############################
def RESULTS (Confusion_Matrix):
        
        Predicted_Accuracy = (Confusion_Matrix[0,0]+Confusion_Matrix[1,1]+Confusion_Matrix[2,2])/Confusion_Matrix.sum()
        print("Predicted_Accuracy:",Predicted_Accuracy)
# Percision = TP/TP+FP
        TP_0 = Confusion_Matrix[0,0]
        FP_0 = Confusion_Matrix[1,0]+Confusion_Matrix[2,0]
        percision_0 = TP_0/(TP_0+FP_0)
        TP_1 = Confusion_Matrix[1,1]
        FP_1 = Confusion_Matrix[0,1]+Confusion_Matrix[2,1]
        percision_1 = TP_1/(TP_1+FP_1)
        TP_2 = Confusion_Matrix[2,2]
        FP_2 = Confusion_Matrix[0,2]+Confusion_Matrix[1,2]
        percision_2 = TP_2/(TP_2+FP_2)
        Predicted_percision = (percision_0+percision_1+percision_2)/3
        print("Predicted_percision",Predicted_percision)
# Specificity = TN/TN+FP
        TN_0 = Confusion_Matrix[1,1]+Confusion_Matrix[1,2]+Confusion_Matrix[2,1]+Confusion_Matrix[2,2]
        Specificity_0 = TN_0/(TN_0+FP_0)
        TN_1 = Confusion_Matrix[0,0]+Confusion_Matrix[0,2]+Confusion_Matrix[2,0]+Confusion_Matrix[2,2]
        Specificity_1 = TN_1/(TN_1+FP_1)
        TN_2 = Confusion_Matrix[0,0]+Confusion_Matrix[0,1]+Confusion_Matrix[1,0]+Confusion_Matrix[1,1]
        Specificity_2 = TN_2/(TN_2+FP_2)
        Predicted_Specificity = (Specificity_0+Specificity_1+Specificity_2)/3
        print("Predicted_Specificity:",Predicted_Specificity)
# Sensitivity =  TP/TP+FN
        TP_0 = Confusion_Matrix[0,0]
        FN_0 = Confusion_Matrix[0,1]+Confusion_Matrix[0,2]
        Sensitivity_0 = TP_0/(TP_0+FN_0)
        TP_1 = Confusion_Matrix[1,1]
        FN_1 = Confusion_Matrix[1,0]+Confusion_Matrix[1,2]
        Sensitivity_1 = TP_1/(TP_1+FN_1)
        TP_2 = Confusion_Matrix[2,2]
        FN_2 = Confusion_Matrix[2,0]+Confusion_Matrix[2,1]
        Sensitivity_2 = TP_2/(TP_2+FN_2)
        Predicted_Sensitivity = (Sensitivity_0+Sensitivity_1+Sensitivity_2)/3
        print("Predicted_Sensitivity",Predicted_Sensitivity)
RESULTS(c_mat)

###########PCA#########
pca = PCA(n_components =4)
pca = pca.fit(train_data_features)

new_train = pca.transform(train_data_features)
new_test = pca.transform(test_data_features)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(new_train, train_data_labels)
predicted_class = clf.predict(new_test)
c_mat_PCA = confusion_matrix(test_data_labels, predicted_class, labels = [0,1,2])
print("Confusion Matrix PCA + D3")
print(c_mat_PCA)
Accuracy_score  = accuracy_score(test_data_labels, predicted_class)
print("Accuracy_score",Accuracy_score)
RESULTS(c_mat_PCA)

########LDA#########            
lda  = LDA()
lda = lda.fit(train_data_features, train_data_labels.ravel())

new_train = lda.transform(train_data_features)
new_test = lda.transform(test_data_features)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(new_train, train_data_labels)
predicted_class = clf.predict(new_test)
c_mat_LDA = confusion_matrix(test_data_labels, predicted_class, labels = [0,1,2])
print("Confusion Matrix LDA + D3")
print(c_mat_LDA)
Accuracy_score  = accuracy_score(test_data_labels, predicted_class)
print("Accuracy_score",Accuracy_score)
################
RESULTS(c_mat_LDA)
