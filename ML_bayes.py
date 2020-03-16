from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#alpha:1.0
#text classify
def nb_demo():
    """
    naive bayes classify the news
    :return:
    """
    #1.obtain dataset
    news =fetch_20newsgroups(subset="all")

    #2.split dataset
    x_train,x_test,y_train,y_test=train_test_split(news.data,news.target)

    #3.feature->tfidf sectortext
    transfer = TfidfVectorizer()
    x_train= transfer.fit_transform(x_train)
    x_test= transfer.transform(x_test)

    #4.naive bayes estimator
    estimator= MultinomialNB()
    estimator.fit(x_train,y_train)

    #5.model evluation
    #method 1)compare true vale and predict value
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare true vale and predict value:\n", y_test == y_predict)

    #method 2)calculate the accuracy
    score=estimator.score(x_test, y_test)
    print("accuracy ratio:\n", score)

    """print("best params:\n", estimator.best_params_)
    print("best score:\n", estimator.best_score_)
    print("best estimator:\n", estimator.best_estimator_)
    print("best cv result:\n", estimator.cv_results_)"""


nb_demo()

