import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

#1.obtian dataset
path="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt"
titanic=pd.read_csv(path)
titanic=titanic.dropna()
#print(titanic)

#2.deal with dataset
#filter feature values and target value
x=titanic[["pclass","age","sex"]]
y=titanic["survived"]
#print(x)
#print(y)
#missing values
x["age"]=x["age"].fillna(lambda x: x.median())
#feature value->dict
x=x.to_dict(orient="records")

#4.split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20)

#5.feature:dictselector
transfer=DictVectorizer()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#6.dession tree
estimator=DecisionTreeClassifier(criterion="entropy",max_depth=8)
estimator.fit(x_train,y_train)

#model evaluation
y_predict=estimator.predict(x_test)
print("y_predict:\n",y_predict)
print("compare:\n",y_test==y_predict)

score=estimator.score(x_test,y_test)
print("accuracy ratio:\n", score)

#visulation
export_graphviz(estimator, out_file="titanic.dot")





























#resouce:http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt