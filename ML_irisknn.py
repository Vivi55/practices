from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#KNN-K nearest neighbor:
#k=1, select the smallest number to define the classify, but easily influenced by outlier
#calculate distance->
#n_neighbours:k value
def knniris_demo():
    """
    knn classify iris
    :return:
    """
    #1.obtain dataset
    iris=load_iris()

    #2.split dataset
    x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,random_state=0)

    #3.Standard Scaler
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.KNN estimator
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    #5.model evaluation
    #method 1)compare true vale and predict value
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare true vale and predict value:\n", y_test==y_predict)

    #method 2)calculate the accuracy
    score=estimator.score(x_test,y_test)
    print("accuracy ratio:\n",score)
    return None

def knn_iris_gscv():
    # cv->cross validation; gs->Grid Search    >>optimization
    """
    knn classify iris + gs +cv
    :return:
    """
    #1.obtain dataset
    iris=load_iris()

    #2.split dataset
    x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,random_state=22)

    #3.Standard Scaler
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.KNN estimator
    estimator= KNeighborsClassifier()

    #add gs+cv
    #prepare parameters
    param_dict = {"n_neignbours:\n": [1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    #5.model evaluation
    #method 1)compare true vale and predict value
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare true vale and predict value:\n", y_test == y_predict)

    #method 2)calculate the accuracy
    score=estimator.score(x_test, y_test)
    print("accuracy ratio:\n", score)

    print("best params:\n", estimator.best_params_)
    print("best score:\n", estimator.best_score_)
    print("best estimator:\n", estimator.best_estimator_)
    print("best cv result:\n", estimator.cv_results_)

    return None




knniris_demo()









