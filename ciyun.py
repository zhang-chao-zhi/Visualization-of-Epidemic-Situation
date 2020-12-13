#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pandas.core.frame import DataFrame
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from os import path
import os,sys
import jieba
import time
import re

#修改路径
route = os.getcwd()
print(route)
os.chdir(r'D:/Program Files (x86)/jupyter/')
route = os.getcwd()
print(route)

#读取数据
f = open("data1.csv","r",encoding='gb18030')
text = f.read()
print(text)
print(type(text))

e = open("data1.csv","r",encoding='gb18030')
stop_words = e.read()
print(stop_words)
print(type(stop_words))

#定义背景图像
background = Image.open("xinxing.jpg")
graph = np.array(background)

#核心代码区域
word_cloud = WordCloud(font_path="simsun.ttc",background_color="white",stopwords=stop_words,mask=graph)
w = word_cloud.generate(text)

# 运用matplotlib展现结果
plt.subplots(figsize=(12,8))
plt.imshow(w)
plt.axis("off")
plt.show()

#对数据通过JIEBA进行分词和统计，可以使用fit_words来生成，第二套方案
words = jieba.lcut(text) #精简模式
print(words)
print(type(words))
list_count =[]
for i in range(0,len(words),1):
    list_count.append(i)
for i in range(0,len(list_count),1):
    list_count[i] =1
c ={"keyword":words,"number":list_count}
data=DataFrame(c)
keyword = pd.pivot_table(data,index=["keyword"],columns=None,values="number",aggfunc=np.sum,margins=False)
keyword.reset_index(level=0,inplace=True)
keyword = keyword.sort_values(by="number",ascending=False) #根据车队编号和货物组合进行排序
print(keyword)

writer = pd.ExcelWriter("词云结果数据集.xlsx")
keyword.to_excel(writer,sheet_name="DATA",index=False) #,endoding="utf_8_sig"
writer.save()

#计算程序运行消耗的时间
print("Running Time: %s Seconds" %time.perf_counter())
print("Well Done!")