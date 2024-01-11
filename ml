PART-A
1> mean mode median
import statistics as st
x=[115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4,90.4]
mean=sum(x)/len(x)
print(mean)
print(st.mean(x))
def median(x):
    x.sort()
    if len(x)%2==0:
        return (x[int(len(x)//2)-1]+x[int((len(x)+1)//2)-1])/2
    else:
        return x[(len(x)+1)//2-1]
med=median(x)
print(med)
print(st.median(x))
y=list(set(x))
mx=0
mode=0
for i in y:
    if x.count(i)>=mx:
        mx=x.count(i)
        mode=i
    else:
        continue   
print(mode)
print(st.mode(x))
var=0
s=0
for i in x:
    s+=(i-mean)**2
var=s/(len(x)-1)
print(var)
print(st.variance(x))
print(var**0.5)
print(st.stdev(x))
from sklearn.preprocessing import MinMaxScaler, StandardScaler
x_r=[[c]for c in x]
mm_sc=MinMaxScaler()
s_sc=StandardScaler()
mms=mm_sc.fit_transform(x_r)
ss=s_sc.fit_transform(x_r)
mms=[v[0] for v in mms]
ss=[v[0] for v in ss]
print(mms)
print(ss)
minv=min(x)
maxv=max(x)
MMS=[]
SS=[]
for i in x:
    MMS.append((i-minv)/(maxv-minv))
    SS.append((i-st.mean(x))/st.stdev(x))
print(MMS)
print(SS)
-------------------------------------------------------
2>linear regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\amana\Downloads\Food-Truck(For Linear Regression Program).csv",names=['a','b'])
x=df['a']
y=df['b']
x_mean=st.mean(x)
y_mean=st.mean(y)
plt.scatter(x,y)
n=0
d=0
for i in range(len(x)):
    n+=(x[i]-x_mean)*(y[i]-y_mean)
    d+=(x[i]-x_mean)**2
m=n/d
c=y_mean-m*x_mean
line=[]
for i in x:
    line.append(m*i+c)
plt.scatter(x,y,label='points',color='r')
plt.plot(x,line,label='line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
nm=st.mean(line)
sse=0
ssr=0
sst=0
for i in range(len(y)):
    sse+=(y[i]-line[i])**2
for i in range(len(line)):
    ssr+=(line[i]-nm)**2
for i in range(len(y)):
    sst+=(y[i]-y_mean)**2
r_sq=1-(sse/sst)
print(sse,ssr,sst,r_sq)
------------------------------------------------------------
3>Decision tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv(r"C:\Users\amana\Downloads\zoo_data(For Decision Tree Program).csv")
df
X=df.drop('1.7',axis=1)
Y=df["1.7"]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
clf_entropy=DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(x_train,y_train)
y_pred=clf_entropy.predict(x_test)
print('confusion matrix: ',confusion_matrix(y_test,y_pred))
print('classification report: ',classification_report(y_test,y_pred))
print('accuracy: ',accuracy_score(y_test,y_pred)*100)
------------------------------------------------------------------------
4>K Mean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
import random as rd
k=3
data=pd.DataFrame({"X1":[5.9, 4.6, 6.2, 4.7, 5.5, 5. , 4.9, 6.7, 5.1, 6.],"X2":[3.2, 2.9, 2.8, 3.2, 4.2, 3. , 3.1, 3.1, 3.8, 3. ]})
print(data["X1"])
X=np.array(list(zip(data["X1"],data["X2"])))
X
cent=np.array(list(zip([6.2, 6.6 ,6.5],[3.2, 3.7, 3.0])))
cent_old=np.zeros(cent.shape)
clusters=np.zeros(len(X))
print(clusters,cent,cent_old)
def eucl(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)
error=eucl(cent,cent_old,None)

iterr=0
while error != 0:
        
        iterr = iterr + 1
        for i in range(len(X)):
            distances = eucl(X[i], cent)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        cent_old = deepcopy(cent)
        print("Old Centroid")
        print(cent)
            
        
        # Finding the new centroids by taking the Mean
        for p in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == p]
            cent[p] = np.mean(points, axis=0)
        print(" New Centroids after ", iterr," Iteration \n", cent)
        error = eucl(cent, cent_old, None)
        print("Error  ... ",error)
        print("Data points belong to which cluster")
        print(clusters)
        print("********************************************************")
------------------------------------------------------------------
5>PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\amana\Downloads\iris(For PCA Program).csv")
x=df.drop('species',axis=1)
y=df['species']
cov=np.cov(x.T)
egval,egvec=np.linalg.eig(cov)
print(egval,egvec)
s_ind=np.argsort(egval)[::-1]
s_egval=egval[s_ind]
s_egvec=egvec[:,s_ind]
eig_sub=s_egvec[:,:2]
x_red=np.dot(eig_sub.transpose(),x.transpose()).transpose()
plt.scatter(x_red[:,0],x_red[:,-1],c=y)
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
PART-B
1> Random Forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv(r"C:\Users\amana\Downloads\pima.csv")
data.head()
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20)
# create the classifier
classifier = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets
classifier.fit(X_train, y_train)
# predicting on the test set
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))
# check Important features
feature_importances_df = pd.DataFrame({"feature": list(X.columns), "importance": classifier.feature_importances_})

# Display
feature_importances_df
x=data.drop(['Outcome','SkinThickness'],axis=1)
y=data['Outcome']
x_tr,x_te,y_tr,y_te=train_test_split(x,y)
classifier.fit(x_tr,y_tr)
y_pred=classifier.predict(x_te)
print("accuracy: ", accuracy_score(y_te,y_pred))
#comparison with decision tree

from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier()
dt.fit(x_tr,y_tr)
y_pr=dt.predict(x_te)
print("accuracy: ",accuracy_score(y_te,y_pr))
-------------------------------------------------------------------------------------------------
2>SVM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV # Import train_test_split function
from sklearn.svm import SVC #Import svm model
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score
data=pd.read_csv(r"C:\Users\amana\Downloads\glass.csv")
data.head()
x = data.drop('Type',axis = 1) 
y = data['Type']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model1=SVC(kernel='sigmoid',gamma=0.001)
model2=SVC(kernel='poly',degree=3)
model3=SVC(kernel='rbf')
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
ypred1=model1.predict(x_test)
ypred2=model2.predict(x_test)
ypred3=model3.predict(x_test)
print(accuracy_score(y_test,ypred1))
print(accuracy_score(y_test,ypred2))
print(accuracy_score(y_test,ypred3))
--------------------------------------------------------------------------------------------
3>Naive BAYES

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,roc_curve
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"C:\Users\amana\Downloads\covid.csv")
le = preprocessing.LabelEncoder()
pc = le.fit_transform(data['pc'].values)
wbc = le.fit_transform(data['wbc'].values)
mc = le.fit_transform(data['mc'].values)
ast = le.fit_transform(data['ast'].values)
bc = le.fit_transform(data['bc'].values)
ldh = le.fit_transform(data['ldh'].values)
y = le.fit_transform(data['diagnosis'].values)
X = np.array(list(zip(pc, wbc, mc, ast, bc, ldh)))
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)
naivee = MultinomialNB()
naivee.fit(xtrain, ytrain)
ypred = naivee.predict(xtest)
print("Accuracy: ", accuracy_score(ytest, ypred))
print("Classification Report: \n", classification_report(ytest, ypred))
lr_probs = naivee.predict_proba(xtest)[:,1]
lr_fpr, lr_tpr, _=roc_curve(ytest, lr_probs)
from matplotlib import pyplot
pyplot.plot(lr_fpr, lr_tpr,label='Naive Bayes Classifier')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
