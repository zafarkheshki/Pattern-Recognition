# Importing libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Open Fisher.txt and read content
csv.register_dialect('myDialect',delimiter = '\t',skipinitialspace=True)

# Read Data
data_table = []
with open('fisher.txt', 'r') as csvFile:
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
# Separate flower types from 1st column and copy whole row to specific type		
data = data[:,1:5]
feature1 = data[:,0]
feature2 = data[:,1]
feature3 = data[:,2]
feature4 = data[:,3]

Mu1 = np.mean(feature1)
Mu2 = np.mean(feature2)
Mu3 = np.mean(feature3)
Mu4 = np.mean(feature4)
# Shift
X1p = feature1-Mu1
X2p = feature2-Mu2
X3p = feature3-Mu3
X4p = feature4-Mu4
# Get covariance matrix
cov_mat = np.cov((X1p,X2p,X3p,X4p))
print("Covariance matrix",cov_mat)
# w - eigenvalues
# v - eigenvectors
w, v = np.linalg.eig(cov_mat)
print("Eigenvalues",w)
print("Eigenvectors",v)
# Sort according to eigenvalues
index = np.argsort(-w)
print("Eigenvalues idx",index)
# Use all eigenvectors for full reconstruction
feature_vector = v[:,index]
print("Feature vector",feature_vector)
RowFeatureVector = np.transpose(feature_vector)
RowZeroMeanData = np.array([X1p, X2p,X3p,X4p])
print("RowZeroMeanData",RowZeroMeanData)
FinalData = np.transpose(np.matmul(RowFeatureVector, RowZeroMeanData))
print("FinalData",FinalData)

# Plotting the data
figure = plt.figure()
axis = figure.add_subplot(111, projection='3d')

axis.scatter(FinalData[:,0],FinalData[:,1],FinalData[:,2], c='r', marker='o')
axis.scatter(feature1,feature2,feature3, c='y', marker='+')
axis.set_xlabel('X Label')
axis.set_ylabel('Y Label')
axis.set_zlabel('Z Label')

plt.show()
