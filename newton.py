#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sympy import *
from future import *
import math
import sys
x=symbols("x")
#f=x*exp(x)-1;  #这个地方的表达式可以根据需要进行更改
#注意到这里，就是先将f对x进行不定积分，然后再求导。
#因为一开始input时候f也不过就是个表达式
f=input("请输入函数的表达式：")
f=integrate(f, x)
f=diff(f)

list_in=input("请输入初始值x,极小值sigma以及想要计算的次数: ").split(" ")
x0=float(list_in[0])
sigma=float(list_in[1])
N=int(list_in[2])

if diff(f).subs(x,x0)==0:
    print(x0)
    sys.exit()
account=1
x_new=x0

while account<=N:
    x_new=x0-f.subs(x,x0)/diff(f).subs(x,x0)
    #x_new=float(x_new)
    if math.fabs(x_new-x0)<sigma:
        print(x_new)
        sys.exit()
    x0=x_new
    account+=1
print("无法解出，不满足条件")


def derivative(coe:list)->list:
    i = len(coe)
    j =1
    newcoe = []
    while j<i:
        newcoe.append(coe[j]*j)
        j=j+1
    return newcoe
def bond(coe:list,s:float)->float:
    i = len(coe)
    bon =0
    while i > 0:
        bon = bon + coe[i - 1] * s ** (i - 1)
        i = i - 1
    return bon

def newton(x:int,coe:list)->int:
    s = x
    i = len(coe)
    d = bond(coe,s)
    while(d>0.00000000000000000000000000000000011):
        list = derivative(coe)
        s = s-(bond(coe,s)/bond(list,s))
        d = bond(coe,s)
        #s=s-(s**2-2*s+1)/(2*s-2)

    return s
#输入要求，输入初始的X0和系数列表，入X^2-2X+1可输入：newton(2,nums) 其中nums=[1,-2,1]
nums=[1,-2,1]
print(newton(2,nums))


