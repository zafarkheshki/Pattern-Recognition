import csv
import numpy as np

csv.register_dialect('myDialect', delimiter='\t', skipinitialspace=True)
# Reading data
data_table = []
with open('Fisher.txt', 'r') as csvFile:
    reader = csv.reader(csvFile, dialect='myDialect')
    data_table = list(reader)
csvFile.close()

header = data_table[0]
del data_table[0]

data = np.zeros((len(data_table), len(data_table[0])))

for i in range(0, len(data_table)):
    tmp = data_table[i]
    for j in range(0, len(tmp)):
        data[i, j] = int(tmp[j])

#  Seprating flower types form data

type0 = data[data[:, 0] == 0, :]
type1 = data[data[:, 0] == 1, :]
type2 = data[data[:, 0] == 2, :]

setosa     = type0
verginica  = type1
versicolor = type2

           ###### T-TEST###########
def T_TEST (class1,class2,Feature):
    class1_X1 = class1[:, Feature]
    class2_X1 = class2[:, Feature]
    
    u1 = np.mean(class1_X1)
    u2 = np.mean(class2_X1)
    
    s2_1 = np.var(class1_X1)
    s2_2 = np.var(class2_X1)
    
    n1 = np.size(class1_X1, 0)
    n2 = np.size(class2_X1, 0)
    print("____________________________________")
    print("Mean:        |   %f %f" % (u1, u2))
    print("Variance     |   %f %f" % (s2_1, s2_2))
    print("Sample size: |   %f %f" % (n1, n2))
    t = (u1 - u2) / (np.sqrt((s2_1 / n1) + (s2_2 / n2)))
    print("T score:     |   %f " % abs(t))
    a = np.power((s2_1 / n1) + (s2_2 / n2), 2)
    b = np.power((s2_1 / n1), 2) / (n1 - 1.0)
    c = np.power((s2_2 / n2), 2) / (n2 - 1.0)
    df = a / (b + c)
    print("Degrees of freedom = df : |  %f " % (df))
    print("____________________________________")
################### t = 2 < T score = 50.5 (According to comparison Petal Length(PL) Good Feature)
print ("T-Test>> Feature: Petal Length(PL)>> Setosa(Type0) and Verginica(Type1)  ")
T_TEST(setosa,verginica,2)
print("From t table, df = 58.6, probability = 0.975, got t value = 2  \n")
print("Result Comparison: t = 2 < T score = 50.5, According to the result we can say that PL is good Feature.\n")
print("--------------------------------------------------------------------------------------------------------")
################### t = 1.98 < T score = 11.4 (According to comparison Petal Length(PL) Good Feature)
print ("T-Test>> Feature: Petal Length(PL)>> Verginica(Type1) and Versicolor(Type2)  ")
T_TEST(verginica,versicolor,2)
print("From t table, df = 97.9, probability = 0.975, got t value = 1.981 \n")
print("Result Comparison: t = 1.98 < T score = 11.4, According to the result we can say that PL is good Feature.\n")
print("--------------------------------------------------------------------------------------------------------")
################### t = 2 < T score = 36.25 (According to comparison Petal Length(PL) Good Feature)
print ("T-Test>> Feature: Petal Length(PL)>> Setosa(Type0) and Versicolor(Type2)  ")
T_TEST(setosa,versicolor,2)
print("From t table, df = 59.2 , probability = 0.975, got t value = 2 \n")
print("Result Comparison: t = 2 < T score = 36.25, According to the result we can say that PL is good Feature.\n")
print("--------------------------------------------------------------------------------------------------------")
################### t = 1.987 < T score = 6.5 (According to comparison Sepal Width(SW) Good Feature)
print ("T-Test>> Feature: Sepal Width(SW)>> Setosa(Type0) and Verginica(Type1)  ")
T_TEST(setosa,verginica,3)
print("From t table, df = 95.5 , probability = 0.975, got t value = 1.987 \n")
print("Result Comparison: t = 1.987  < T score = 6.5, According to the result we can say that SW is good Feature.\n")
print("--------------------------------------------------------------------------------------------------------")
################### t = 1.983 < T score = 3.3 (According to comparison Sepal Width(SW) Good Feature)
print ("T-Test>> Feature: Sepal Width(SW)>> Verginica(Type1) and Versicolor(Type2)  ")
T_TEST(verginica,versicolor,3)
print("From t table, df = 97.9 , probability = 0.975, got t value = 1.983 \n")
print("Result Comparison: t = 1.983 < T score = 3.3, According to the result we can say that SW is good Feature.\n")
print("--------------------------------------------------------------------------------------------------------")
#################### t = 1.99 < T score = 9.6 (According to comparison Sepal Width(SW) Good Feature)
print ("T-Test>> Feature: Sepal Width(SW)>>Setosa(Type0) and Versicolor(Type2)  ")
T_TEST(setosa,versicolor,3)
print("From t table, df = 94.7 , probability = 0.975, got t value = 1.99  \n")
print("Result Comparison: t = 1.99 < T score = 9.6, According to the result we can say that SW is good Feature.\n")
print("--------------------------------------------------------------------------------------------------------")



##################################   ANOVA-TEST   ########################################
print("\n")
print("   ANOVA-TEST    ")


type0 = data[data[:, 0] == 0, 1:]
type1 = data[data[:, 0] == 1, 1:]
type2 = data[data[:, 0] == 2, 1:]

def ANOVA_TEST (select_feature):

    g0= type0[:,select_feature]
    g1= type1[:,select_feature]
    g2= type2[:,select_feature]
    # Global mean
    g_all = np.array([g0,g1,g2])
    # Group mean
    g0_u = np.mean(g0)
    g1_u = np.mean(g1)
    g2_u = np.mean(g2)
    print("Mean",g0_u,g1_u,g2_u)
    print('Set up hypotheses and determine level of significance')
    print("H0:",g0_u,g1_u,g2_u)
    print('Means are not all equal')
    # Degree of freedom

    k = 3                          # Three groups
    N = g_all.size                 # Total amount of the data

    # Degree of freedom
    df1 = k-1                      
    df2 = N-k                     
    critical_value = 3.05          #  From F distribution table, alpha = 0.05

    print('k:',k, "\nN:", N, '\ncritical value:', critical_value)
    # SSB
    g012_u = np.array([g0_u,g1_u,g2_u])
    g_size = np.array([g0.size, g1.size, g2.size])
    global_u = np.mean(g_all)
    print('group Size:', g_size)
    print('global mean:', global_u)
    ssb = np.sum(np.multiply(g_size, np.power(np.subtract(g012_u, global_u),2)))
    print('Sum of Squares Error = SSB:',ssb)

    # SSE
    sse_g0 = np.sum(np.power(np.subtract(g0,g0_u),2))             
    sse_g1 = np.sum(np.power(np.subtract(g1,g1_u),2))
    sse_g2 = np.sum(np.power(np.subtract(g2,g2_u),2))
    print('Group Error:',sse_g0,sse_g1,sse_g2)
    sse = sse_g0 + sse_g1 + sse_g2
    print('Sum of Squares Error = SSE:',sse)

    # Between group means squares
    ms1 = ssb/df1
    # Error means squares
    ms2 = sse/df2
    # F value
    F = ms1/ms2
    print('Between Group Means Squares:', ms1)
    print('Error Means Squares:', ms2)
    print('F:', F)
    print("--------------------------------")
    print('Comparison:',F, '>',critical_value, ' This feature is good to distinguish three types flowers')
    print("--------------------------------\n")
# Comparison: 878.76 > 3.06, According to comparison PW is a good feature to distinguish three types flowers
print("ANOVA-TEST >> Feature: Petal Width(PW)")
print("--------------------------------------")
ANOVA_TEST(0)
# Comparison: 1061.5 > 3.06, According to comparison PL is a good feature to distinguish three types flowers
print("ANOVA-TEST >> Feature: Petal Width(PL)")
print("--------------------------------------")
ANOVA_TEST(1)
# Comparison: 49.8 > 3.06, According to comparison SW is a good feature to distinguish three types flowers
print("ANOVA-TEST >> Feature: Sepal Width(SW)")
print("--------------------------------------")
ANOVA_TEST(2)
# Comparison:  118.5 > 3.06, According to comparison SL is a good feature to distinguish three types flowers
print("ANOVA-TEST >> Feature: Sepal Length(SL)")
print("--------------------------------------")
ANOVA_TEST(3)




