import sklearn.model_selection
import numpy
import copy

all_data = numpy.loadtxt("flats_moscow_mod.txt", delimiter='\t', skiprows=1)
data_train, data_test = sklearn.model_selection.train_test_split(all_data, test_size=0.3, random_state=3)
print(data_train.shape, data_test.shape)

X_train = copy.copy(data_train[:,1:])
y_train = copy.copy(data_train[:,0])

X_test = data_test[:,1:]
y_test = data_test[:,0]

n = X_train.shape[0]
c = numpy.concatenate((X_train, numpy.ones(X_train.shape[0]).reshape(-1, 1)), axis=1)
a = numpy.dot(numpy.dot(sla.inv(numpy.dot(c.T, c)), c.T), y_train)
print('общая площадь квартиры' + str(a[0]))
print('жилая площадь квартиры' + str(a[1]))
print('площадь кухни' + str(a[2]))
print('расстояние от центра' + str(a[3]))
print('расстояние до метро в минутах' + str(a[4]))

print(data_train[0])
poly = X_train


for i in range(n):
    for j in range(5):
        poly[i][j] = poly[i][j] * a[j]

otv = 0

for i in range(n):
    sum = 0
    for j in range(5):
        sum = sum + poly[i][j]
    sum = (sum - y_train[i]) * (sum - y_train[i])
    otv = otv + sum
        
otv = otv / n
print(otv)


import math

Z_train = copy.copy(data_train[:,1:])

n = Z_train.shape[0]

for i in range(n):
    for j in range(3, 5):
        if Z_train[i][j] < 1:
            Z_train[i][j] = 1
        Z_train[i][j] = math.sqrt(Z_train[i][j])
        
print(Z_train[0])
        
cc = numpy.concatenate((Z_train, numpy.ones(Z_train.shape[0]).reshape(-1, 1)), axis=1)
aa = numpy.dot(numpy.dot(sla.inv(numpy.dot(cc.T, cc)), cc.T), y_train)

print('общая площадь квартиры' + str(aa[0]))
print('жилая площадь квартиры' + str(aa[1]))
print('площадь кухни' + str(aa[2]))
print('расстояние от центра' + str(aa[3]))
print('расстояние до метро в минутах' + str(aa[4]))


poll = copy.copy(Z_train)

for i in range(n):
    for j in range(5):
        poll[i][j] = poll[i][j] * aa[j]

otv = 0

#print(poll[0])


for i in range(n):
    sum = 0
    for j in range(5):
        sum = sum + poll[i][j]
    sum = (sum - y_train[i]) * (sum - y_train[i])
    otv = otv + sum
        
otv = otv / n
print(otv)


data = numpy.loadtxt('train.txt', delimiter=',')
data_test = numpy.loadtxt('test.txt', delimiter=',')

print(data.shape)
data_val, data_train = sklearn.model_selection.train_test_split(data, test_size=0.5, random_state=3)
X_val = copy.copy(data_val[:,0])
y_val = copy.copy(data_val[:,1])

X_train = copy.copy(data_train[:,0])
y_train = copy.copy(data_train[:,1])

X_test = copy.copy(data_test[:,0])
y_test = copy.copy(data_test[:,1])
print(X_val.shape)

n = X_train.shape[0]

def make_po(alpha, XX):
    n = len(XX)
    poly = numpy.empty([n, alpha])
    poly[:,alpha - 2] = XX
    for i in range(alpha):
        poly[:,i] = poly[:,alpha - 2]
        poly[:, i] = numpy.power(poly[:, i], alpha - 1 - i)
    return poly


def dsp(alpha, n, lmbd):
    X_train = copy.copy(data_train[:,0])
    poly = make_po(alpha, X_train)
    ans = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly) + lmbd*numpy.eye(7)), poly.T), y_train)

        
    poly = make_po(alpha, X_val)
    res = sklearn.metrics.mean_squared_error(y_val, numpy.dot(poly, ans))
    
    return res

def find_res(alpha, n, lmbd):
    X_train = copy.copy(data_train[:,0])
    poly = make_po(alpha, X_train)
    ans = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly) + lmbd*numpy.eye(7)), poly.T), y_train)

        
    poly = make_po(alpha, X_test)
    res = sklearn.metrics.mean_squared_error(y_test, numpy.dot(poly, ans))
    
    print(res)


    
i = 0
j = 0
was = 100
best = 0
while i < 1:
    i = i + 0.0005
    new = dsp(7, n, i)
    if (was > new):
        was = new
        best = i
        
print(was)
print("best lambda is: " + str(best))

find_res(7, n, best)

    
poly = make_po(7, X_train)
print(sla.det(numpy.dot(poly.T, poly)))
print(sla.det(numpy.dot(poly.T, poly) + 0.0885 * numpy.eye(7)))

poly = make_po(7, X_train)
ans1 = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly)), poly.T), y_train)
ans2 = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly) + 0.0885*numpy.eye(7)), poly.T), y_train)
fig = plt.figure(figsize=(16, 8))

def y1(x, ans1, poly):
    sum = 0
    cur = 1
    for j in range(6, -1, -1):
        sum = sum + cur * ans1[j]
        cur = cur * x
    return sum

x_plot = numpy.linspace(-1, 2, 1000)

plt.plot(x, y1(x, ans1, poly), 'r', label='no regularization')
plt.plot(x, y1(x, ans2, poly), 'g', label='regularization')
plt.plot(X_train, y_train, 'yo', label='Train data')
plt.plot(X_test, y_test, 'bo', label='Test data')
plt.plot(X_val, y_val, 'co', label='Val data')
  
plt.axis([-0.45, 1.5, 0, 9])
plt.legend(loc='upper left')

plt.grid()
plt.show()



