KMEAN----
import numpy as np
import pandas as pd
from copy import deepcopy
k=3
import random as rd
import matplotlib.pyplot as plt
X = pd.read_csv('kmeans.csv')
print(X)
X = X[["X1","X2"]]
#Visualise data points
plt.scatter(X["X1"],X["X2"],c='black')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()
x1 = X['X1'].values
x2 = X['X2'].values
x1
x2
X = np.array(list(zip(x1, x2)))
print(X)
C_x = [6.2, 6.6 ,6.5]
C_y = [3.2, 3.7, 3.0]
Centroid = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids")
print(Centroid.shape)
Centroid
type(Centroid)
Centroid_old = np.zeros(Centroid.shape)
print(Centroid_old)
clusters = np.zeros(len(X))
print(clusters)
def euclidean(a,b, ax=1):
    return np.linalg.norm(a-b, axis=ax)
error = euclidean(Centroid, Centroid_old,None)
print(error)
iterr = 0
while error != 0:
        # Assigning each value to its closest cluster
        iterr = iterr + 1
        for i in range(len(X)):
            #print("Data Points")
            #print(X[i])
            distances = euclidean(X[i], Centroid)
            #print("Distances")
            #print(distances)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        Centroid_old = deepcopy(Centroid)
        print("Old Centroid")
        print(Centroid_old)
            
        
        # Finding the new centroids by taking the Mean
        for p in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == p]
            Centroid[p] = np.mean(points, axis=0)
        print(" New Centroids after ", iterr," Iteration \n", Centroid)
        error = euclidean(Centroid, Centroid_old, None)
        print("Error  ... ",error)
        print("Data points belong to which cluster")
        print(clusters)
        print("********************************************************")
X = pd.read_csv('kmeans.csv')
X = X[["X1","X2"]]
#Visualise data points
plt.scatter(X["X1"],X["X2"],c=clusters)
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()
--------------------------------------------
Mean mode------
x=[115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4,90.4]
mean = sum(x) / len(x)
mean
import statistics
print(statistics.mean(x))
x.sort()
x
n=len(x)
if  n% 2 == 0:
    median1 = x[n//2]
    median2 = x[n//2 - 1]
    median = (median1 + median2)/2
else:
    median = x[n//2]
print("Median is: " + str(median))
print(statistics.median(x))
frequency = {}
for value in x:
        frequency[value] = frequency.get(value, 0) + 1
frequency
frequencs=list(frequency.values())
print(frequencs)
most_frequent = max(frequencs)
most_frequent
modes = [key for key, value in frequency.items()
                      if value == most_frequent]
modes
print(statistics.mode(x))
variance = sum((i - mean) ** 2 for i in x) / (n-1)
variance
print(statistics.variance(x))
standarddev= variance ** 0.5
standarddev
print(statistics.stdev(x))
import numpy as np
max = np.max(x)
min = np.min(x)
scaled_arr = np.array([(i - min) / (max - min) for i in x])
print(scaled_arr)
x=[[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(x)
print(scaled)
X=[115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4,90.4]
for i in range(len(X)):
        X[i] = (X[i] - statistics.mean(X)) /(statistics.stdev(X))
X=[115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4,90.4]
stand_arr = np.array([(i -statistics.mean(X) ) / (statistics.stdev(X)) for i in X])
stand_arr
print(round(statistics.mean(stand_arr),2))
from numpy import asarray
from sklearn.preprocessing import StandardScaler
# define data
data = asarray([[100, 0.001],
[8, 0.05],
[50, 0.005],
[88, 0.07],
[4, 0.1]])
print(data)
# define standard scaler
scaler = StandardScaler()
# transform 

scaled = scaler.fit_transform(data)
print(scaled)
print(round(np.mean(scaled),5))
print(np.std(scaled))
----------------------------------------
Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/vikas/Desktop/Food-Truck.csv")
class Mocd:
    
    def mean_(self, x):
        sum = 0
        for ele in x:
            sum += ele
        mean = sum / len(x)
        return mean
    
    def standard_deviation(self, x):
        sd = 0
        m = self.mean_(x)
        for xi in x:
            sd += (xi - m)**2
        sd = sd/len(x)
        return sd
    
    def slope_(self,x,y):
        x_mean = self.mean_(x)
        y_mean = self.mean_(y)
        
        numerator = 0
        denominator = 0
        
        for xi, yi in zip(x,y):
            numerator += (xi-x_mean)*(yi-y_mean)
            denominator += (xi-x_mean)**2
        slope = 0
        slope = numerator/denominator
        return slope
    
    def intercept_(self, x,y):
        x_mean = self.mean_(x)
        y_mean = self.mean_(y)
        
        slope = self.slope_(x,y)
        intercept = y_mean - (slope * x_mean)
        return intercept
    
    def ssr_(self, x, y):
        ssr = 0
        y_pred = self.primitive_linear_regression(x,y)
        for yi, yi_cap in zip(y, y_pred):
            ssr += (yi - yi_cap)**2
        return ssr
    
    def sst_(self, x, y):
        sst = 0
        y_mean = self.mean_(y)
        for yi in y:
            sst += (yi - y_mean)**2
        return sst
    
    def r2_(self, x, y):
        r2 = 0
        ssr = self.ssr_(x,y)
        sst = self.sst_(x,y)
        
        r2 = 1 - (ssr / sst)
        return r2
    
    def primitive_linear_regression(self, x,y):
        y_pred = []
        m = self.slope_(x, y)
        c = self.intercept_(x, y)
        for xi in x:
            y_pred.append((m*xi) + c)
        return y_pred
    
    def summary_(self, x, y):
        print("-"*55)
        print("X Mean :", self.mean_(x))
        print("y Mean :", self.mean_(y))
        print("")
        print("X Standard Deviation :", self.standard_deviation(x))
        print("y Standard Deviation :", self.standard_deviation(y))
        print("")
        print("Slope :", self.slope_(x,y))
        print("Intercept :", self.intercept_(x,y))
        print("Sum squared error :", self.ssr_(x,y))
        print("Sum squared total :", self.sst_(x,y))
        print("coefficient of determination - R2 :", self.r2_(x,y))
        print("-"*55)        
df.head()
X = df['Attribute']
y = df['Label']
obj = Mocd()
res = obj.r2_(X,y)
res
obj.summary_(X,y)
plt.scatter(X, y)
plt.plot(X, obj.primitive_linear_regression(X, y), c = 'red')
plt.show()
------------------------------------------
Logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Log_reg.csv")
df.head()
df['Target'].value_counts()
list(X.columns)
class Logistic_Regression:
    
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.loss_list = []
        self.w1_list = []
        self.w2_list = []
        self.b_list = []
        self.y_pred_list = []
        
    def cost_function(self, target, attribute1, attribute2):
        cost = 0
        m = len(target)
        
        for y_actual, x1, x2 in zip(target, attribute1, attribute2):
            if (1 - self.sigmoid(x1, x2)) > 0 and self.sigmoid(x1, x2) > 0:
                cost += (y_actual * math.log(self.sigmoid(x1, x2))) + ((1 - y_actual) * math.log(1 - self.sigmoid(x1, x2)))
            elif (1 - self.sigmoid(x1, x2)) < 0 and self.sigmoid(x1, x2) > 0:
                cost += (y_actual * math.log(self.sigmoid(x1, x2)))
            elif (1 - self.sigmoid(x1, x2)) > 0 and self.sigmoid(x1, x2) < 0:
                cost += ((1 - y_actual) * math.log(1 - self.sigmoid(x1, x2)))
        
        cost_function = -(1 / m) * cost
        self.loss_list.append(cost_function)
        
        # print("Loss :", cost_function)    
        
    def linear_eqn(self, x1, x2):
        y = self.w1 * x1 + self.w2 * x2 + self.b
        return y
    
    def sigmoid(self, x1, x2):
        y = self.linear_eqn(x1, x2)
        
        try:
            y_pred = 1 / (1+ math.exp(-y))
        except OverflowError:
            y_pred = 1 / (1+ math.exp(-700))
        
        
        return y_pred
    
    def weights(self, alpha, target, attribute1, attribute2, iters):
        
        error1 = 0
        error2 = 0
        error3 = 0
        m = len(target)
        
        for y_actual, x1, x2 in zip(target, attribute1, attribute2):
            y_pred = self.sigmoid(x1, x2)
            
            error1 += (y_pred - y_actual) * x1
            error2 += (y_pred - y_actual) * x2
            error3 += (y_pred - y_actual)
        old_loss = 0
        loss = 0
        
        while iters != 0:
            
            
            self.w1 = self.w1 - (alpha / m) * error1
            self.w2 = self.w2 - (alpha / m) * error2       
            self.b = self.b - (alpha / m) * error3  
        
            self.w1_list.append(self.w1)
            self.w2_list.append(self.w2)
            self.b_list.append(self.b)
            old_loss = loss
            loss = self.cost_function(target, attribute1, attribute2)
            iters -= 1
    
    def custom_compile(self, X, y, iters = 100, alpha = 0.1):
        target = y
        attribute1 = X['Attribute 1']
        attribute2 = X['Attribute 2']
        
        
        self.weights( alpha, target, attribute1, attribute2, iters)
        
    def accuracy(self, y_actual, X):
        m = len(y_actual)
        correct = 0
        
        for x1, x2 in zip(X['Attribute 1'], X['Attribute 2']):
            y_pred = self.sigmoid(x1, x2)
            self.y_pred_list.append(y_pred)
            
        for i,j in zip(y_actual, self.y_pred_list):
            if i == j:
                correct += 1

        print("Accuracy :", correct/m)
        return correct/m
    
    def predict(self, X):
        prediction = self.sigmoid(X[0], X[1])
        print(prediction)
obj = Logistic_Regression(1, 1, 1)


y = df['Target']
X = df.drop('Target', axis = 1)


obj.custom_compile(X, y, 1000, alpha = 0.001)
plt.plot(obj.loss_list)
plt.plot(obj.w1_list)
plt.plot(obj.w2_list)
plt.show()
obj.accuracy(y,X)
obj.y_pred_list
obj.predict([34.623660	,78.024693])
