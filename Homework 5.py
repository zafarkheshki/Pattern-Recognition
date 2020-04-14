# Including libraries
import csv
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csv.register_dialect('myDialect',delimiter = '\t',skipinitialspace=True)

# Read data
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
		    
X1 = data[data[:,0] == 0,1:]
X2 = data[data[:,0] == 1,1:]
X3 = data[data[:,0] == 2,1:]

figure = plt.figure()
axis = figure.add_subplot(111, projection='3d')

axis.scatter(X1[:,0], X1[:,1], X1[:,2], c='r', marker='o')
axis.scatter(X2[:,0], X2[:,1], X2[:,2], c='b', marker='^')
axis.scatter(X3[:,0], X3[:,1], X3[:,2], c='c', marker='x')

axis.set_xlabel('X Label')
axis.set_ylabel('Y Label')
axis.set_zlabel('Z Label')
plt.title('Original Data')
plt.show()
# Create a 2D array
Mu1 = np.array([np.mean(X1, axis=0)])
Mu2 = np.array([np.mean(X2, axis=0)])
Mu3 = np.array([np.mean(X3, axis=0)])

Mu1 = Mu1.transpose()
Mu2 = Mu2.transpose()
Mu3 = Mu3.transpose()

Mu = (Mu1+Mu2+Mu3)/3.0

S1 = np.cov((X1[:,0], X1[:,1],X1[:,2],X1[:,3]))
S2 = np.cov((X2[:,0], X2[:,1],X2[:,2],X2[:,3]))
S3 = np.cov((X3[:,0], X3[:,1],X3[:,2],X3[:,3]))

# Within class scatter matrix
Sw = S1 + S2 + S3


r,N1 = X1.shape
r,N2 = X2.shape
r,N3 = X3.shape
m1 = Mu1-Mu
m2 = Mu2-Mu
m3 = Mu3-Mu
Sb1 = N1*np.matmul(m1,m1.transpose())
Sb2 = N2*np.matmul(m2,m2.transpose())
Sb3 = N3*np.matmul(m3,m3.transpose())
# Between-class scatter matrix
Sb = Sb1 + Sb2 + Sb3;

# LDA projection
SwSb = np.matmul(np.linalg.pinv(Sw),Sb)

# Projection vector
w, v = np.linalg.eig(SwSb)
# w - eigenvalues
# v - eigenvectors
print("Eigenvalues")
print(w)
print("Eigenvectors")
print(v)
# Sorting according to eigenvalues
index = np.argsort(-w)
print("Eigenvalues idx")
print(index)
# Eigenvectors for full reconstruction
W1 = v[:,index]
print("Weight vector")
print(W1)

# Project data samples along he projection axes
new_X1 = np.matmul(X1,W1)
new_X2 = np.matmul(X2,W1)
new_X3 = np.matmul(X3,W1)

fig_t = plt.figure()
axis_t = fig_t.add_subplot(111, projection='3d')

axis_t.scatter(new_X1[:,0], new_X1[:,1], new_X1[:,2], c='r', marker='o')
axis_t.scatter(new_X2[:,0], new_X2[:,1], new_X2[:,2], c='b', marker='^')
axis_t.scatter(new_X3[:,0], new_X3[:,1], new_X3[:,2], c='c', marker='x')

axis_t.set_xlabel('X Label')
axis_t.set_ylabel('Y Label')
axis_t.set_zlabel('Z Label')
plt.title('Transformed Data')
plt.show()

