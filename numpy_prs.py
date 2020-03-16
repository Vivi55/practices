import numpy as np

#build array,list

def array_demo01():
    array=np.array([[1,2,3],
                    [2,3,4,]])

    print(array)
    print('number of dim:',array.ndim)
    print('shape:',array.shape)
    print('size:',array.size)

def array_demo02():
    a=np.array([2,23,4],dtype=np.int)
    #print(a.dtype)
    b=np.zeros((2,3),dtype=int)
    #print(b)
    c=np.arange(10,20,2)
    #print(c)
    d=np.arange(12).reshape((3,4))
    #print(d)
    e=np.linspace(1,10,6).reshape((2,3))
    print(e)

def array_demo03():
 #caculation
    a=np.array([10,20,30,40])
    b=np.arange(4)
    #print(a,b)
    c=a-b
    e=b**2
    #print(c,e)
    #print(b<3)
    m=np.array([[1,3],
                [3,4]])
    n=np.arange(4).reshape((2,2))
    x=m*n
    y=np.dot(m,n)
    #print(x,y)

    v=np.random.random((2,4))
    print(v)
    print('sum:\n',np.sum(v))
    print(np.sum(v,axis=1))#行数求和
    print(np.sum(v,axis=0))#列数求和

    print('min:\n',np.min(v))
    print('max:\n',np.max(v))

def array_demo04():
    #index
    A=np.arange(2,14).reshape((3,4))
    print(A)
    print(np.argmin(A))
    print(np.argmax(A))
    print(np.mean(A))
    print(np.average(A))
    print(np.median(A))#中心值
    print(np.cumsum(A))#累加值
    print(np.diff(A))#累差
    print(np.nonzero(A))#非零数
    print(np.sort(A))#排序
    print(np.transpose(A))#反向
    print(A.T)#反向
    print((A.T).dot(A))
    print(np.clip(A,5,9))#小于5改为5，大于9改为9
    print(np.mean(A,axis=0))

def array_demo04():
    B=np.arange(3,15).reshape(3,4)
    print(B)
    print(B[2])
    print(B[2,1])
    print(B[2][1])
    print(B[2,:])
    print(B[:,1])
    print('---------------')
    for row in B:
        print(row)
    print('---------------')
    for column in B.T:
        print(column)
    print('---------------')
    print(B.flatten())
    print('---------------')
    for item in B.flat:
        print(item)

def array_demo05():
    C=np.array([1,1,1])[:,np.newaxis]
    D=np.array([2,2,2])[:,np.newaxis]
    E=print(np.vstack((C,D)))#vertical stack
    F=print(np.hstack((C,D)))#horizontal stack
    X=print(np.concatenate((C,D,D,C),axis=0))
    Y=print(np.concatenate((C, D, D, C), axis=1))
    print(C[:,np.newaxis])#row->column

def array_demo06():
    A=np.arange(12).reshape(3,4)
    print(A)
    #print(np.split(A,2,axis=1))#纵向分割，必须等量
    #print(np.split(A,3,axis=0))#横向分割，必须等量
    #print(np.array_split(A,3,axis=1))#不等量的分割
    print(np.vsplit(A,3))
    print(np.hsplit(A,4))

def array_demo07():
    #赋值
    a= np.arange(4)
    print(a)
    b=a
    c=a
    d=b
    a[0]=21
    print(a)
    print(b is a)
    print(c is a)
    print(d is a)
    d[1:3]=[22,23]
    print(d)
    print('----------------')
    b=a.copy()#deep copy
    print(b)
    a[3]=6666
    print(a)
    print(b)


#array_demo01()
#array_demo02()
#array_demo03()
#array_demo04()
#array_demo05()
#array_demo06()
array_demo07()






