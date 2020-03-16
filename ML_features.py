from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def datasets_demo():
    """
    use sklearn datasets
    :return:
    """
    #obtain dataset
    iris=load_iris()
    print("iris dataset:\n",iris)
    print("dataset describe:\n", iris["DESCR"])
    print("feature names:\n", iris.feature_names)
    print("featrue:\n", iris.data, iris.data.shape)

    #split dataset
    x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,test_size=0.3, random_state=22)
    print("x_train:\n",x_train,x_train.shape)

    return None

def dict_demo():
    """
    dictionary features
    :return:
    """
    data=[{'city':'London','temperature':50},{'city':'Auckland','temperature':100},{'city':'Paris','temperature':150},]
    #1.instance tranfer
    tranfer=DictVectorizer(sparse=False)# make sure matrix

    #2.fit_transform()
    data_new =tranfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("feature names:\n",tranfer.get_feature_names())

    return None

def count_demo():
    """
    CountVectorizer
    :return:
    """
    data=["my life is brilliant","my love is pure","I saw an angle of that i am sure"]
    #1.instance transfer
    transfer =CountVectorizer(stop_words=("is","am","of","that"))#stop_words delete words

    #2.fit_tranform
    data_new=transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray()) # toarray for matrix
    print("feature names:\n",transfer.get_feature_names())

    return None

def spl_words(text):
    """
    split Chinese words "每一个不曾起舞的日子，都是对生命的一种辜负。"->"每 一个 不曾 起舞 的 日子，都 是 对 生命 的 一种 辜负。"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))# generator->list->string

def count_Chiness_demo():
    """
    auto split words->jieba
    :return:
    """
    data=["每一个不曾起舞的日子，都是对生命的一种辜负。",
          "别做思想上的巨人，行动上的矮子。",
          "你没有如期归来，而这正是离别的意义。"]
    data_new=[]
    for sent in data:
        data_new.append(spl_words(sent))
    print(data_new)
    #1.instance transfer
    transfer =CountVectorizer(stop_words=["一个","正是", "不曾","一种"])

    #2.fit_transform
    data_final=transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("feature names:\n",transfer.get_feature_names())

    return None

#TF-IDF(term frequency-inverse document frequency)
def tfidf_demo():
    """
    TF-IDF for text
    :return:
    """
    data = ["每一个不曾起舞的日子，都是对生命的一种辜负。",
            "别做思想上的巨人，行动上的矮子。",
            "你没有如期归来，而这正是离别的意义。"]
    data_new = []
    for sent in data:
        data_new.append(spl_words(sent))
    print(data_new)
    # 1.instance transfer
    transfer = TfidfVectorizer(stop_words=["一个", "正是", "不曾", "一种"])

    # 2.fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("feature names:\n", transfer.get_feature_names())

    return None


if __name__=='__main__':
#    count_demo()
#    dict_demo()
#    datasets_demo()

#    print(spl_words("每一个不曾起舞的日子，都是对生命的一种辜负。"))
#    count_Chiness_demo()
    tfidf_demo()










