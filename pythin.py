import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,1,0.05)
#print(x),
#y=sin(2*pi*x)
y=np.sin(2*np.pi*x)
#print(y),
plt.plot(x,y,'b--*',label='sin')
plt.title('My First Plot')
plt.xlabel('x label')
plt.ylabel('y label')
plt.legend(loc='best')
plt.show()
fig,ax=plt.subplots(2,2)
ax[0,1].plot(x,y)
plt.show()
#y2=np.cos(2*pi*x)
y2=np.cos(2*np.pi*x)
#print(y2),
fig,ax=plt.subplots()
ax.plot(x,y,'g--*',label='sin')
ax.plot(x,y2,'r--o',label='cos')
ax.set(title='My First Plot')
legend=ax.legend(loc='upper center')
plt.show()
#保存
fig.savefig('myfig.jpg')
#读取数据
import pandas as pd
df=pd.read_csv('data.csv',index_col='年份')
df.head()
x=df.index.values
y=df['人均GDP（元）'].values
from pylab import mpl

mpl.rcParams['font.sans-serif']=['FangSong']
#指定默认字体
fig,ax=plt.subplots()
ax.plot(x,y,'r--*')
ax.set(title='人均GDP走势图',xlabel='年份')
plt.show()
fig,ax=plt.subplots()
ax.pie(y[:5],labels=x[:5])
plt.show()
fig,ax=plt.subplots()
ax.pie(y[:5],labels=x[:5],explode=[0,0.05,0.1,0.15,0.2])
plt.show()
#绘制词云图
with open('data1.csv',encoding='gb18030') as file:
    words=file.read()
#print(words)
from wordcloud import WordCloud
wordcloud=WordCloud(font_path='C:/Windows/Fonts/simfang.ttf',
                   background_color='white',width=600,height=600,max_words=30).generate(words)
image=wordcloud.to_image()
image.show()
#绘制指定形状的词云图
from PIL import Image
images=Image.open('DITU.jpg')
maskImages=np.array(images)
wordcloud=WordCloud(font_path='C:/Windows/Fonts/simfang.ttf',
                   background_color='white',width=400,height=400,max_words=40,mask=maskImages).generate(words)
image=wordcloud.to_image()
image.save('xinxing2.jpg')
image.show()

#pandas中的绘图函数
import pandas as pd

df=pd.read_csv('data.csv',index_col='年份')
df.head()
x=df.index.values
y=df['人均GDP（元）'].values
from pylab import mpl

mpl.rcParams['font.sans-serif']=['FangSong']
import matplotlib.pyplot as plt

fig,ax=plt.subplots()
ax.plot(x,y,'r--o')
ax.set(title='人均GDP走势图',xlabel='年份')
plt.show()

#另外一种写法
fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(x,y,'r--o')
ax.set(title='人均GDP走势图',xlabel='年份')
plt.show()

#pandas中的绘图函数
df['人均GDP（元）'].plot(color='r')
plt.show()
df['人均GDP（元）'].plot(kind='bar',color='skyblue')
plt.show()
df.plot()
plt.show()
df.plot(kind='bar')
plt.show()
df.plot(kind='bar',stacked=True)
plt.show()
#散点图及三维散点图
import numpy as np

data=np.random.randint(0,100,size=[40,40])
data
x,y=data[0],data[1]

ax=plt.subplot(111)
ax.scatter(x[:10],y[:10],c='r')
ax.scatter(x[10:20],y[10:20],c='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()
#三维散点图
from mpl_toolkits.mplot3d import Axes3D

x,y,z=data[0],data[1],data[2]
ax=plt.subplot(111,projection='3d')
ax.scatter(x[:10],y[:10],z[:10],c='b')
ax.scatter(x[10:20],y[10:20],z[10:20],c='r')
ax.scatter(x[30:40],y[30:40],z[30:40],c='g')

ax.set_xlabel('X')
ax.set_zlabel('Z')
ax.set_ylabel('Y')

plt.show()

#绘制词云图及jieba分词
from wordcloud import WordCloud
import jieba

print(list(jieba.cut("我是一个段落，怎么绘制词云图？")))
with open('paragraph.txt',encoding='utf-8') as file:
    text=file.read()
print(text)
wordlist_jieba=jieba.cut(text)

wordlist='/'.join(wordlist_jieba)
print(wordlist)
wordcloud=WordCloud(font_path='C:/Windows/Fonts/simfang.ttf',background_color='white',width=400,height=400,max_words=40,mask=maskImages).generate(wordlist)
image=wordcloud.to_image()
image.save('xinxing3.jpg')
image.show()
#散点图矩阵
from sklearn.datasets import load_iris
iris_dataset=load_iris()
#print(iris_dataset.DESCR)
#绘制散点图矩阵
import pandas as pd

iris_df=pd.DataFrame(iris_dataset['data'],columns=iris_dataset.feature_names)
iris_df.head()
grr=pd.plotting.scatter_matrix(iris_df,c=iris_dataset['target'],figsize=(15,15),marker='o',alpha=0.5)
#逻辑回归中的sigmoid函数
#matplotlib,numpy

def sigmoid(x):
    return 1/(1+np.exp(-x))
from pylab import mpl

mpl.rcParams['axes.unicode_minus']=False
x=np.arange(-5,6,1)
fig,ax=plt.subplots()
ax.plot(x,sigmoid(x),'r')