import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#1.obtain dataset
data=pd.read_csv("~~")
#2.deal data
    #a. reduce the range of data
data=data.query("x<2.5&x>2.0&y<1.5&y>1.0")
            #2.0<x<2.5
            #1.0<y<1.5

    #b. time->specific
time_value=pd.to_datatime(data["time"],unit="s")
print(time_value.values)
date=pd.DatatimeIndex(time_value)
data["day"]=date.day
data["weekday"]=date.weekday
data["hour"]=date.hour

    #c.filter the few checkin places
place_count=data.groupby("place_id").count()["row.id"]
place_count[place_count>3]
data_final=data["place_id"].isin(place_count[place_count>3].index.values)
    #feature value  + target value
x=data_final[["x","y","accuracy","day","weekday","hour"]]
y=data_final["place_id"]

#3.standard scaler
x_train, x_test, y_train, y_test=train_test_split(x,y)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#4.knn estimetor
estimator = KNeighborsClassifier()

#5.model selection & optimization
param_dict = {"n_neignbours:\n": [3,5,7,9]}
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






























