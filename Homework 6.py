import csv
import numpy as np 

csv.register_dialect('myDialect',delimiter = '\t',skipinitialspace=True)
# Reading data
data_table = []
# Opening File Data from Source
with open('Fisher.txt', 'r') as csvFile:
	reader = csv.reader(csvFile, dialect='myDialect')
	data_table = list(reader)
csvFile.close()

header = data_table[0]
del data_table[0]
data = np.zeros((len(data_table), len(data_table[0])))

# Reading Data from file
for i in range(0,len(data_table)):
	tmp = data_table[i]
	for j in range(0,len(tmp)):
		data[i,j] = int(tmp[j])
        
# Split class
x0 = data[data[:,0] == 0]
x1 = data[data[:,0] == 1]
x2 = data[data[:,0] == 2]

# Get index for splitting data 80% - 20%
data_split = 0.8
samples_x0 = round(len(x0)*data_split)
samples_x1 = round(len(x1)*data_split)
samples_x2 = round(len(x2)*data_split)

# Make train data set
train_x0 = x0[0:samples_x0,:]
train_x1 = x1[0:samples_x1,:]
train_x2 = x2[0:samples_x2,:]

train_data = np.concatenate((train_x0,train_x1, train_x2))

test_x0 = x0[samples_x0:,:]
test_x1 = x1[samples_x1:,:]
test_x2 = x2[samples_x2:,:]

test_data = np.concatenate((test_x0,test_x1, test_x2))
# Get class mean value
MuX0= np.mean(train_x0[:,1:], axis=0)
MuX1= np.mean(train_x1[:,1:], axis=0)
MuX2= np.mean(train_x2[:,1:], axis=0)
print(MuX0)
# L2 norm
acu = 0
acu2 = 0
for i in range(0, test_data.shape[0]):
    # L2 norm
    s = test_data[i,1:] #Sample
    # Euclidean distance
    dist1 = np.sqrt(np.power(s[0]-MuX0[0], 2) + np.power(s[1]-MuX0[1], 2) + np.power(s[2]-MuX0[2], 2) + np.power(s[3]-MuX0[3], 2))
    dist2 = np.sqrt(np.power(s[0]-MuX1[0], 2) + np.power(s[1]-MuX1[1], 2) + np.power(s[2]-MuX1[2], 2) + np.power(s[3]-MuX1[3], 2))
    dist3 = np.sqrt(np.power(s[0]-MuX2[0], 2) + np.power(s[1]-MuX2[1], 2) + np.power(s[2]-MuX2[2], 2) + np.power(s[3]-MuX2[3], 2))

    d = np.array([dist1, dist2, dist3])
    idx = np.argsort(d)
    predicted_class = idx[0]

    if(test_data[i,0] == predicted_class):
        acu = acu+1
    # Manhattan distance
    dist4 = np.absolute(s[0]-MuX0[0]) + np.absolute(s[1]-MuX0[1]) + np.absolute(s[2]-MuX0[2]) + np.absolute(s[3]-MuX0[3])
    dist5 = np.absolute(s[0]-MuX1[0]) + np.absolute(s[1]-MuX1[1]) + np.absolute(s[2]-MuX1[2]) + np.absolute(s[3]-MuX1[3])
    dist6 = np.absolute(s[0]-MuX2[0]) + np.absolute(s[1]-MuX2[1]) + np.absolute(s[2]-MuX2[2]) + np.absolute(s[3]-MuX2[3])
    d2 = np.array([dist4, dist5, dist6])

    idx2 = np.argsort(d2)
    predicted_class2 = idx2[0]
    
    if(test_data[i,0] == predicted_class2):
        acu2 = acu2+1   
        
print("\n")        
print("**************************  TASK 1   ***************************")
print("Accuracy for euclidean distance: %f" %(acu/float(test_data.shape[0])))
print("Accuracy for Manhattan distance: %f" %(acu2/float(test_data.shape[0])))
print("******************************************************************")
print("\n")   

###########################################
#               TASK2                     #
###########################################

np.random.seed(1222)
np.random.shuffle(data)

data_split = 0.8
samples_training = round(data.shape[0]*data_split)

train_data = data[0:samples_training,:]
test_data = data[samples_training:,:]

# Note: Data is saved as row vector
X1 = train_data[:,1]
X2 = train_data[:,2]
X3 = train_data[:,3]
X4 = train_data[:,4]

Mu1 = np.mean(X1)
Mu2 = np.mean(X2)
Mu3 = np.mean(X3)
Mu4 = np.mean(X4)

# Shift
X1_train = X1-Mu1
X2_train = X2-Mu2
X3_train = X3-Mu3
X4_train = X4-Mu4

# For test data
# Note: Data is saved as row vector
X1_test = test_data[:,1]
X2_test = test_data[:,2]
X3_test = test_data[:,3]
X4_test = test_data[:,4]

Mu1_test = np.mean(X1_test)
Mu2_test = np.mean(X2_test)
Mu3_test = np.mean(X3_test)
Mu4_test = np.mean(X4_test)

# Shift
X1_test = X1_test-Mu1_test
X2_test = X2_test-Mu2_test
X3_test = X3_test-Mu3_test
X4_test = X4_test-Mu4_test

# Get covariance matrix
cov_mat = np.cov((X1_train,X2_train,X3_train,X4_train))
# w - eigenvalues
# v - eigenvectors
w, v = np.linalg.eig(cov_mat)

# Sort according to eigenvalues
index = np.argsort(-w)

# Use all eigenvectors for full reconstruction
feature_vector = v[:,index]

RowFeatureVector = np.transpose(feature_vector)
RowZeroMeanData = np.array([X1_train, X2_train, X3_train, X4_train])
RowZeroMeanData_test = np.array([X1_test, X2_test, X3_test, X4_test])

FinalData = np.transpose(np.matmul(RowFeatureVector, RowZeroMeanData))
FinalData_test = np.transpose(np.matmul(RowFeatureVector, RowZeroMeanData_test))

# Split class
# For training data
final_x0 = FinalData[train_data[:,0] == 0,:]
final_x1 = FinalData[train_data[:,0] == 1,:]
final_x2 = FinalData[train_data[:,0] == 2,:]

# Get class mean value
final_MuX0= np.mean(final_x0, axis=0)
final_MuX1= np.mean(final_x1, axis=0)
final_MuX2= np.mean(final_x2, axis=0)
# L2 norm
final_acu = 0
final_acu2 = 0

for i in range(0, FinalData_test.shape[0]):
    # L2 norm
    sample = FinalData_test[i,:] # Sample
    # Euclidean distance
    distance1 = np.sqrt(np.power(sample[0]-final_MuX0[0], 2) + np.power(sample[1]-final_MuX0[1], 2) + np.power(sample[2]-final_MuX0[2], 2) + np.power(sample[3]-final_MuX0[3], 2))
    distance2 = np.sqrt(np.power(sample[0]-final_MuX1[0], 2) + np.power(sample[1]-final_MuX1[1], 2) + np.power(sample[2]-final_MuX1[2], 2) + np.power(sample[3]-final_MuX1[3], 2))
    distance3 = np.sqrt(np.power(sample[0]-final_MuX2[0], 2) + np.power(sample[1]-final_MuX2[1], 2) + np.power(sample[2]-final_MuX2[2], 2) + np.power(sample[3]-final_MuX2[3], 2))

    final_distance = np.array([distance1, distance2, distance3])
    final_idx = np.argsort(final_distance)
    final_predicted_class = final_idx[0]

    if(test_data[i,0] == final_predicted_class):
        final_acu = final_acu+1

    # Manhattan distance
    distance4 = np.absolute(sample[0]-final_MuX0[0]) + np.absolute(sample[1]-final_MuX0[1]) + np.absolute(sample[2]-final_MuX0[2]) + np.absolute(sample[3]-final_MuX0[3])
    distance5 = np.absolute(sample[0]-final_MuX1[0]) + np.absolute(sample[1]-final_MuX1[1]) + np.absolute(sample[2]-final_MuX1[2]) + np.absolute(sample[3]-final_MuX1[3])
    distance6 = np.absolute(sample[0]-final_MuX2[0]) + np.absolute(sample[1]-final_MuX2[1]) + np.absolute(sample[2]-final_MuX2[2]) + np.absolute(sample[3]-final_MuX2[3])
    
    final_distance2 = np.array([distance4, distance5, distance6])
    final_idx2 = np.argsort(final_distance2)
    final_predicted_class2 = final_idx2[0]
    
    if(test_data[i,0] == final_predicted_class2):
        final_acu2 = final_acu2+1   

print("**************  TASK 2   *********************")

print("Accuracy for euclidean distance: %f" %(final_acu/float(test_data.shape[0])))
print("Accuracy for Manhattan distance: %f" %(final_acu2/float(test_data.shape[0])))
print("***********************************************")
