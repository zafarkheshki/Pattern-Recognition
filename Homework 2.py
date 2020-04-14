#########################################

import numpy as np
import matplotlib.pyplot as plt

# Reading file
My_file = open("Fisher.txt")
content = list(My_file)

#########################################
# Separating  first row (column names) from  data set.

Names = content[0].split()
features = Names[1:] # Returns string with features' names
# print (features)

strings = []                                              # Creates an empty array
for string in content[1:]:                                # To put each element from index features into array,
    strings.append(string.split())                        # Adds space between them
print (string)
#Step to convert obtain stings for future plotting

for string in strings:
    for i in range(len(string)):                # String to integer 
        string[i] = int(string[i])              

# Empty array for 3 classes
character0 = []
character1 = []
character2 = []

# Data seperation according to its classes

for string in strings:
   if string[0] == 0:
       character0.append(string[1:])
   elif string[0] == 1:
       character1.append(string[1:])
   elif string[0] == 2:
       character2.append(string[1:])

# Saving the data to numpy array before plotting
character0 = np.array(character0)
character1 = np.array(character1)
character2 = np.array(character2)

###########################################
# Plotting of each flower index on x axis and its  features on y axis

for i in range (len(features)):
    fig, ax = plt.subplots()
    ax.plot(range(len(character0)), character0[:,i], 'o', label="Setosa")
    ax.plot(range(len(character1)), character1[:,i], '+', label="Verginica")
    ax.plot(range(len(character2)), character2[:,i], '*', label="Versicolor")
    ax.set_xlabel("Index of each flower")
    ax.set_ylabel(features[i])
    fig.suptitle(features[i])
    ax.legend()
    plt.show()

###########################################
# Merging data into one set

Data_set = np.concatenate((character0,character1,character2), axis = 0)
print(Data_set)

# Tags matching with data set, where 0-49 = 0;

tags = np.zeros((150,1))
tags[50:100] = 1                  # 50-99 = 1
tags[100:150] = 2                 # 100-149 = 2

# In this step predicted-variable is created
predicted_tag = np.ones((150,1))*7

# Separating the data based on the figures

for index, data in enumerate(Data_set):
    if data[0] <= 5:
        predicted_tag [index] = 0
    elif data[0] >= 17:
        predicted_tag [index] = 1
    else:
        predicted_tag [index] = 2


###########################################

# Calculating how many correct tags been gotten form prediction

C_Predictions = 0
for i in range(len(tags)):
    if predicted_tag [i] == tags[i]:
        C_Predictions += 1
        
print(C_Predictions)

# Calculating the accuracy of the system

accuracy = C_Predictions / 150 * 100
print("accuracy = " ,accuracy, "%")






