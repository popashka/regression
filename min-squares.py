import numpy
import sklearn.metrics
import scipy.linalg as sla
import matplotlib.pyplot as plt
import copy


data_train = numpy.loadtxt('train.txt', delimiter=',')
data_test = numpy.loadtxt('test.txt', delimiter=',')X_train = data_train[:,0]
y_train = data_train[:,1]


X_test = data_test[:,0]
y_test = data_test[:,1]

X_train = data_train[:,0]
y_train = data_train[:,1]


X_test = data_test[:,0]
y_test = data_test[:,1]

n = X_train.shape[0]
b = numpy.ones(n)

c = numpy.concatenate((X_train.reshape(-1, 1), b.reshape(-1, 1)), axis=1)
a = numpy.dot(numpy.dot(sla.inv(numpy.dot(c.T, c)), c.T), y_train)
print(str(a[0]) + 'x + ' + str(a[1]))


x = numpy.linspace(-0.5,1.5,10)
plt.title('Graph of y=kx+b, Plots of samples')

plt.plot(x, x * a[0] + a[1], 'b', label='prediction')


plt.plot(X_train, y_train, 'ro', label='train data')
plt.plot(X_test, y_test, 'yo', label='text data')
plt.legend(loc='upper left')
plt.grid()
plt.show()

def step(alpha):
    poly = numpy.empty([21, alpha])
    poly[:,alpha - 2] = X_train
    for i in range(alpha):
        poly[:,i] = poly[:,alpha - 2]
        poly[:, i] = numpy.power(poly[:, i], alpha - 1 - i)
    ans = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly)), poly.T), y_train)
    for i in range(alpha - 1, -1, -1):
        print str(ans[i]) + "x^" + str(alpha - i - 1),
        if i > 0:
            print "+",
        if i == 0:
            print ""
    
step(13)


def graph0(alpha):
    X_train = data_train[:,0]
    poly = numpy.empty([21, alpha])
    poly[:,alpha - 2] = X_train
    for i in range(alpha):
        poly[:,i] = poly[:,alpha - 2]
        poly[:,i] = numpy.power(poly[:, i], alpha - 1 - i)
    ans = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly)), poly.T), y_train)

    y = numpy.empty([21, 1])

    for i in range(21):
        sum = 0
        for j in range(alpha):
            sum = sum + poly[i][j] * ans[j]
        y[i][0] = sum
    plt.plot(X_train, y, label='prediction')


        
graph0(13)


plt.plot(X_train, y_train, 'ro', label='train data')
plt.plot(X_test, y_test, 'yo', label='test data')
plt.axis([-1, 2, 0, 8])

plt.legend(loc='upper left')
plt.grid()
plt.show()


def make_poly(alpha, n, X):
    poly = numpy.empty([n, alpha])
    poly[:,alpha - 2] = X
    for i in range(alpha):
        poly[:,i] = poly[:,alpha - 2]
        poly[:, i] = numpy.power(poly[:, i], alpha - 1 - i)
    return poly


def disp(alpha, n):
    X_train = copy.copy(data_train[:,0])
    poly = make_poly(alpha, n, X_train)
    X_train = copy.copy(data_train[:,0])
    ans = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly)), poly.T), y_train)
    print("Test " + str(alpha - 1))
    print("Train " + str(sklearn.metrics.mean_squared_error(y_train, numpy.dot(poly, ans))))

        
    X_test = copy.copy(data_test[:,0])
    poly = make_poly(alpha, n, X_test)

    print("Test " + str(sklearn.metrics.mean_squared_error(y_test, numpy.dot(poly, ans))))
    
            

        
for i in range(1, 11):
    disp(i, n)

fig = plt.figure(figsize=(20, 10))


def graph(alpha):
    X_train = data_train[:,0]
    poly = numpy.empty([21, alpha])
    poly[:,alpha - 2] = X_train
    for i in range(alpha):
        poly[:,i] = poly[:,alpha - 2]
        poly[:,i] = numpy.power(poly[:, i], alpha - 1 - i)
    ans = numpy.dot(numpy.dot(sla.inv(numpy.dot(poly.T, poly)), poly.T), y_train)

    y = numpy.empty([21, 1])

    for i in range(21):
        sum = 0
        for j in range(alpha):
            sum = sum + poly[i][j] * ans[j]
        y[i][0] = sum
    if alpha == 2:
        plt.plot(X_train, y, 'c', label='ax + b')
    if alpha == 3:
        plt.plot(X_train, y, 'b', label='ax^2+bx+c')
    if alpha == 4:
        plt.plot(X_train, y, 'g', label='ax^3+..');
    if alpha == 5:
        plt.plot(X_train, y, 'm', label='ax^4+..')
    if alpha == 7:
        plt.plot(X_train, y, 'r', label='ax^6+..')
  
graph(2)
graph(3)
graph(4)
graph(5)
graph(7)



plt.plot(X_train, y_train, 'ro', label='train data')
plt.plot(X_test, y_test, 'yo', label='test data')
plt.axis([-0.5, 1.5, 3, 8])

plt.legend(loc='upper left')
plt.grid()
plt.show()