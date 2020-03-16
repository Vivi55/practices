from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import pandas as pd

def minmax_demo():
    """
    normalization:data in [0,1]
    Max & Min are outliers
    :return:
    """
    #1.obtain dataset
    data = pd.read_csv("iris.txt")
    data=data.iloc[:,:3]
    print("data:\n", data)
    #2.instance transfer
    transfer=MinMaxScaler(feature_range=[2,3])
    #3.fit_tranform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

#(x-mean)/std-> depend on degree of concentration
def std_demo():
    """
    standard
    :return:
    """
    # 1.obtain dataset
    data = pd.read_csv("iris.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)
    # 2.instance transfer
    transfer = StandardScaler()
    # 3.fit_tranform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None

def variance_demo():
    """
    filter low variance feature
    :return:
    """
    #1. obtain the dataset
    data=pd.read_csv("~~")
    #print(data)
    data=data.iloc[:,1:-2] #confirm the range of dataset

    #2. instance transfer
    transfer=VarianceThreshold(threshold=5)

    #3. fit_transfer
    data_new=transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)

#Pearson correlation coefficient->[-1,1]
#r>0, positive correlation; r<0, negative correlation;  |r|=1,same ;r=0,no correlation
#|r|<0.4,low correlation;0.4<|r|<0.7,significant correlation;|r|<1, high correlation
    #calulate two variables correlation coefficient
    r=pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("correlation coefficient:\n",r)

    return None


if __name__=="__main__":
    minmax_demo()
#    std_demo()
#    variance_demo()





