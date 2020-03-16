from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import pandas as pd

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

#5.estimator
estimator=RandomForestClassifier()
param_dict = {"n_estimators:\n": [120,200,300,500,800,1200],"max_depth":[5,8,15,25,30]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
estimator.fit(x_train, y_train)

#6.model eveluation
#method 1
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("compare true vale and predict value:\n", y_test == y_predict)
#method 2
score=estimator.score(x_test, y_test)
print("accuracy ratio:\n", score)

print("best params:\n", estimator.best_params_)
print("best score:\n", estimator.best_score_)
print("best estimator:\n", estimator.best_estimator_)
print("best cv result:\n", estimator.cv_results_)

#random->train :feature+target :bootstrap[1,2,3,4,5]
