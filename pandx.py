# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:41:21 2020

@author: Administrator
"""

# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx