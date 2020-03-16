from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz

#criterion->entropy----gini
#max_depth->less
def desstree_iris():
    """
    classify iris by dessiontree
    :return:
    """
    #1.obtain dataset
    iris= load_iris()

    #2.split dataset
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=22)

    #3.feature(creative)
    #4.dession tree estimator
    estimator=DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)

    #5.model evluation
    #1)method:
    y_predict= estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("compare true vale and predict value:\n",y_predict==y_test)

    #2)method:
    score=estimator.score(x_test,y_test)
    print("accuracy ratio:\n", score)

    # visualization
    export_graphviz(estimator,out_file="iris_tree.dot",feature_names=iris.feature_names)#load in feature names
    #open the iris_tree.dot->copy contxt->open the web:Webgraphviz
    #paste ->done
    #clear feature names: load in names


    return None

#cannnot use for complicated dataset
#how to solve?
#cart()->reduce branches
#random forest()




desstree_iris()














