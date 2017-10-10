import numpy as np
from numpy.lib import recfunctions as rfn
import random
import matplotlib.pyplot as plt
from sklearn import svm

# func to modify columns with string to integers
def AddColumns(arr, columnName):
    types = np.unique(arr[columnName])

    for type in types:
        arr = rfn.append_fields(arr, names=columnName + "_" + type, data=-np.ones(N), usemask=False)

    for r in arr:
        for type in types:
            if r[columnName] == type:
                r[columnName + '_' + type] = 1
                break

    return rfn.drop_fields(arr, drop_names=columnName)

# read csv file
csvFile = 'HR_comma_sep.csv'
data = np.genfromtxt(csvFile, delimiter=',', names=True, dtype="f8, f8, i8, i8, i8, i8, i8, i8, S10, S6")
names = data.dtype.fields.keys()

# check if there exists null values
for name in names:
    if data[name].dtype == 'float64' or data[name].dtype == 'int64':
        print np.isnan(data[name]).any(), name

# add salary_type and sales_type columns
data = AddColumns(data, "salary")
data = AddColumns(data, "sales")

# split train data 80% and test data 20%:
N = len(data)
N80 = int(N * 0.8)

x = range(1,N)
random.shuffle(x)

data_train = data[x[0:N80]]
data_test = data[x[N80 + 1:N]]

#
Y = data_train["left"]
temp = rfn.drop_fields(data_train, drop_names="left")
X = []
for i in range(0, N80):
    arr = []
    for j in range(0, len(temp[0])):
        arr.append(temp[i][j])
    X.append(arr)
X = np.array(X)


# Run SVM
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(X, Y)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)



rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
#names = data.dtype.fields.keys()
