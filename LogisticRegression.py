import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("Social_Network_Ads.csv")
print(dataset)
X=dataset.iloc[:,0:2]
print(X)
y=dataset.iloc[:,-1:]
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.25,random_state=42)
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
# print(train_X)
train_X=SS.fit_transform(train_X)
test_X=SS.transform(test_X)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(train_X,train_y)
y_predict=classifier.predict(test_X)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_y,y_predict)
print(cm)
# from mlxtend.plotting import plot_confusion_matrix
# fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(6,6),cmap=plt.cm.Greens)
# plt.scatter(test_X,test_y ,color='red')
# plt.plot(test_X,y_predict,color='blue')
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()
import pickle
pickle.dump(SS,open('scalar.pkl','wb'))
ssc=pickle.load(open('scalar.pkl','rb'))
pickle.dump(classifier,open('lr_classifier.pkl','wb'))
model=pickle.load(open('lr_classifier.pkl','rb'))