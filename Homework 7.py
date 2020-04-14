import csv
import numpy as np 
import matplotlib.pyplot as plt

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
# Spilt class		
X1 = data[data[:,0] == 0,:]
X2 = data[data[:,0] == 1,:]
X3 = data[data[:,0] == 2,:]
		
data_split = 0.8

print(X1.shape[0])
# Get index for splitting the data
samples_X1 = int(X1.shape[0]*data_split)
samples_X2 = int(X2.shape[0]*data_split)
samples_X3 = int(X3.shape[0]*data_split)
print(samples_X1)
# Make train dataset
train_X1 = X1[0:samples_X1,:]
train_X2 = X2[0:samples_X2,:]
train_X3 = X3[0:samples_X3,:]

print(train_X1.shape)
train_data = np.concatenate((train_X1, train_X2, train_X3), axis=0)

# Make test dataset
test_X1 = X1[samples_X1:,:]
test_X2 = X2[samples_X2:,:]
test_X3 = X3[samples_X3:,:]

print(test_X1.shape)

test_data = np.concatenate((test_X1, test_X2, test_X3), axis=0)


X1 = train_data[train_data[:,0] == 0,1:5]
X2 = train_data[train_data[:,0] == 1,1:5]
X3 = train_data[train_data[:,0] == 2,1:5]


# Create a 2D array
Mu1 = np.array([np.mean(X1, axis=0)])
Mu2 = np.array([np.mean(X2, axis=0)])
Mu3 = np.array([np.mean(X3, axis=0)])

Mu1 = Mu1.transpose()
Mu2 = Mu2.transpose()
Mu3 = Mu3.transpose()

Mu = (Mu1+Mu2+Mu3)/3.0

S1 = np.cov((X1[:,0], X1[:,1], X1[:,2], X1[:,3]))
S2 = np.cov((X2[:,0], X2[:,1], X2[:,2], X2[:,3]))
S3 = np.cov((X3[:,0], X3[:,1], X3[:,2], X3[:,3]))
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

# Sort according to eigenvalues
index = np.argsort(-w)

# Use all eigenvectors for full reconstruction
W1 = v[:,index]

# Reduce dimension
W1[:,3] = 0

# Project data samples along the projection axes
new_X1 = np.matmul(X1,W1)
new_X2 = np.matmul(X2,W1)
new_X3 = np.matmul(X3,W1)


# Prject TEST DATA
test_project = np.matmul(test_data[:,1:5],W1)

#get class mean value for prjected data
MuX1 = np.mean(new_X1, axis = 0)
MuX2 = np.mean(new_X2, axis = 0)
MuX3 = np.mean(new_X3, axis = 0)

acu = 0
acu2 = 0 
# Predict class
for i in range(0, test_data.shape[0]):
    #L2 norm
    s = test_project[i,:] #sample
    
    # Euclidean distance
    dist1 = np.sqrt(np.power(s[0]-MuX1[0], 2) + np.power(s[1]-MuX1[1], 2) + np.power(s[2]-MuX1[2], 2))
    dist2 = np.sqrt(np.power(s[0]-MuX2[0], 2) + np.power(s[1]-MuX2[1], 2) + np.power(s[2]-MuX2[2], 2))
    dist3 = np.sqrt(np.power(s[0]-MuX3[0], 2) + np.power(s[1]-MuX3[1], 2) + np.power(s[2]-MuX3[2], 2))

    d = np.array([dist1, dist2, dist3])
    idx = np.argsort(d)
    predicted_class = idx[0]
    
    if(test_data[i,0] == predicted_class):
        acu = acu+1
    
    # Manhattan distance
    dist4 = np.absolute(s[0]-MuX1[0]) + np.absolute(s[1]-MuX1[1]) + np.absolute(s[2]-MuX1[2])
    dist5 = np.absolute(s[0]-MuX2[0]) + np.absolute(s[1]-MuX2[1]) + np.absolute(s[2]-MuX2[2])
    dist6 = np.absolute(s[0]-MuX3[0]) + np.absolute(s[1]-MuX3[1]) + np.absolute(s[2]-MuX3[2])

    d2 = np.array([dist4, dist5, dist6])
    idx2 = np.argsort(d2)
    predicted_class2 = idx2[0]
    
    if(test_data[i,0] == predicted_class2):
        acu2 = acu2+1   
print("Accuracy for euclidean distance: %f" %(acu/float(test_project.shape[0])))
print("Accuracy for Manhattan distance: %f" %(acu2/float(test_project.shape[0])))
