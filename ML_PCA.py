from sklearn.decomposition import PCA

def pca_demo():
    """
    PCA decomposition(Dimensionality reduction)
    :return:
    """
    #1.obtain dataset
    data=[[2,4,8,5],[3,5,8,9],[4,5,7,1]]

    #2. instance transform
    #tranfer=PCA(n_components=2)#n_components: Integer: reduced dimension; Decimal: Percent of information retained
    tranfer = PCA(n_components=0.95)
    #3.fit_transform
    data_new=print(tranfer.fit_transform(data))

    return None

if __name__=="__main__":
    pca_demo()

