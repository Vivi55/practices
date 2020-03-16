import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demo01():
    s=pd.Series([1,3,5,7,np.nan,99,4])
    #print(s)

    dates=pd.date_range('20200301',periods=6)
    #print(dates)

    df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
    df1=pd.DataFrame(np.arange(12).reshape(3,4))
    #print(df,'\n',df1)

    df2 = pd.DataFrame({'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo'})
    print(df2)
    print(df2.dtypes)
    print(df2.index)
    print('---------------------------------------------------')
    print(df2.columns)
    print(df2.values)
    print(df2.describe())
    print(df2.T)
    print('---------------------------------------------------')
    print(df.sort_index(axis=1,ascending=False))#倒着排序
    print(df.sort_index(axis=0,ascending=False))#倒着排序
    print('---------------------------------------------------')
    print(df2.sort_values(by='E'))#根据E列排序

def select_demo02():
    #select dataset
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    print(df['A'],'\n',df.A)
    print(df[0:3],df['20130102':'20130104'])#选择列&行
    #select by label:loc
    print(df.loc['20130102'])
    print(df.loc['20130102':,['A','B']])
    #select by position:iloc
    print(df.iloc[3:5,1:3])
    print(df.iloc[[1,3,5],1:3])
    #mixed selection:ix
    #print(df.ix[:3,['A','C']])
    #Boolean indexing
    print(df[df.A<8])

def set_demo03():
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    df.iloc[2,2]=1111
    #print(df)
    df.loc['20130101','B']=2222
    #print(df)
    df.B[df.A>4]=0
    #print(df)
    df['F']=np.nan
    #print(df)
    df['E']=pd.Series([1,2,3,4,5,6],index=pd.date_range('20130101',periods=6))
    print(df)

def missing_demo():
    #建立一个丢失data的环境
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    df.iloc[0, 1] = np.nan
    df.iloc[1, 2] = np.nan
    print(df)

    #print(df.dropna(axis=0,how='any'))
    print(df.fillna(value=0))
    print(df.isnull())#检验是否有缺失值
    print(np.any(df.isnull())==True)

def red_demo():
    data=pd.read_csv('~~')

def hebing():
    #concatenating
    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
    print(df1)
    print(df2)
    print(df3)
    res=pd.concat([df1,df2,df3],axis=1,ignore_index=True)
    #print(res)
def hebing_demo02():
    #join,['inner','outer']
    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
    print(df1)
    print(df2)
    #res1=print(pd.concat([df1,df2],join='outer'))
    #res2=print(pd.concat([df1,df2],join='inner',ignore_index=True))
    #join_axes
    #res3=print(pd.concat([df1,df2],axis=1,join_axes=[df1.index]))
    res4=print(df1.append(df2,ignore_index=True))
    #添加一行数据
    s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
    res5=print(df1.append(s1,ignore_index=True))

def hebing_demo03():
    left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})
    right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})
    print(left)
    print(right)
    res=print(pd.merge(left,right,on='key'))

def hebing04():
    left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                         'key2': ['K0', 'K1', 'K0', 'K1'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})
    right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                          'key2': ['K0', 'K0', 'K0', 'K0'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})

    print(left)
    print(right)
    #how=['left','right','outer','inner']
    #res=print(pd.merge(left,right,on=['key1','key2'],how='outer'))

def hebing5():
    #indicator
    df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
    df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
    print(df1)
    print(df2)

    # 依据col1进行合并，并启用indicator=True，最后打印出
    #res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)

    #give the indicator a custom name
    res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
    print(res)

def hebing05():
    #merged by index
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                         'B': ['B0', 'B1', 'B2']},
                        index=['K0', 'K1', 'K2'])
    right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                          'D': ['D0', 'D2', 'D3']},
                         index=['K0', 'K2', 'K3'])
    print(left)
    print(right)
    res=pd.merge(left,right,left_index=True,right_index=True,how='outer')
    print(res)

def overlapping_demo():
    boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
    girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
    print(boys)
    print(girls)
    res=print(pd.merge(boys,girls,on='k',suffixes=['_boy','_girl'],how='outer'))

    #merge is as same as join

def visual_demo():
    #plot data
    #Series
    data = pd.Series(np.random.randn(1000), index=np.arange(1000))
    data=data.cumsum()

    #plt.plot(x=,y=)
    #DataFrame
    data=pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list("ABCD"))

    print(data.head(3))
    data = data.cumsum()


#select_demo02()
#set_demo03()
#missing_demo()
#hebing()
#hebing_demo02()
#hebing_demo03()
#hebing04()
#hebing5()
#hebing05()
#overlapping_demo()
#visual_demo()
#plot methods:
# bar, hist, box, kde, area, scatter, hexbin,pie
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
ax=data.plot.scatter(x='A',y='B',color='Black',label='Class 1')
data.plot.scatter(x='A',y='C',color='Pink',label='Class 2',ax=ax)
#data.plot()
plt.show