from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

def linear1():
    """LR optimization predict the price of Boston houses"""
    #1.obatin dataset
    boston=load_boston()

    #2.split dataset
    x_train,x_test,y_train,y_test=train_test_split(boston.data, boston.target,random_state=22)

    #3.feature: normalization
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.estimator
    #fit()model
    estimator= LinearRegression()
    estimator.fit(x_train,y_train)
    #coef_intercept
    print("coef:\n",estimator.coef_)
    print("intercept:\n",estimator.intercept_)


    #save model
    joblib.dump(estimator,"my_LR.pkl")
    #load model
    estimator=joblib.load("my_LR.pkl")

    #5.model evaluation
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("LR error:\n", error)

    return None

def linear2():
    """SGD optimization predict the price of Boston houses"""
    #1.obatin dataset
    boston=load_boston()

    #2.split dataset
    x_train,x_test,y_train,y_test=train_test_split(boston.data, boston.target,random_state=22)

    #3.feature: normalization
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.estimator
    #fit()model
    estimator= SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=100000)
    estimator.fit(x_train,y_train)
    #coef_intercept
    print("coef:\n",estimator.coef_)
    print("intercept:\n",estimator.intercept_)

    # 5.model evaluation
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    error=mean_squared_error(y_test,y_predict)
    print("SGD error:\n",error)

    return None
#how to know which method is better-->>#5.model evaluation
#MSE(Mean Squared Error)
#result:LR is better if SDR do not be changed params

def linear3():
    """Ridge optimization predict the price of Boston houses"""
    # 1.obatin dataset
    boston = load_boston()

    # 2.split dataset
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3.feature: normalization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.estimator
    # fit()model
    estimator = Ridge(alpha=0.5)
    estimator.fit(x_train, y_train)
    # coef_intercept
    print("coef:\n", estimator.coef_)
    print("intercept:\n", estimator.intercept_)

    # 5.model evaluation
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("Ridge error:\n", error)

    return None

linear1()















