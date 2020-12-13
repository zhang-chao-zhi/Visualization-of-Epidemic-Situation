#!/usr/bin/env python
# -*- coding:utf-8 -*-
import bar_chart_race as bcr

# 如果出现SSL错误,则全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 获取数据
df = bcr.load_dataset('covid19_tutorial')
# print(df)

# 生成GIF图像
bcr.bar_chart_race(df, 'covid19_horiz.gif')

# orientation='v',生成柱状图
bcr.bar_chart_race(df, 'covid19_horiz.gif', orientation='v')


# 设置排序方式,asc-升序
bcr.bar_chart_race(df, 'covid19_horiz.gif', sort='asc')

# 设置最多能显示的条目数,6条
bcr.bar_chart_race(df, 'covid19_horiz.gif', n_bars=6)

# 选取如下5个国家的数据
bcr.bar_chart_race(df, 'covid19_horiz.gif', fixed_order=['Iran', 'USA', 'Italy', 'Spain', 'Belgium'])


# 设置数值的最大值，固定数值轴
bcr.bar_chart_race(df, 'covid19_horiz.gif', fixed_max=True)


# 图像帧数。数值越小，越不流畅。越大，越流畅。
bcr.bar_chart_race(df, 'covid19_horiz.gif', steps_per_period=3)


# 设置20帧的总时间，此处为200ms
bcr.bar_chart_race(df, 'covid19_horiz.gif', steps_per_period=20, period_length=200)

# 输出MP4
bcr.bar_chart_race(df, 'covid19_horiz.mp4', interpolate_period=True)


# figsize-设置画布大小，默认(6, 3.5)
# dpi-图像分辨率，默认144
# label_bars-显示柱状图的数值信息，默认为True
# period_label-显示时间标签信息，默认为True
# title-图表标题
bcr.bar_chart_race(df, 'covid19_horiz.gif', figsize=(5, 3), dpi=100, label_bars=False,
                   period_label={'x': .99, 'y': .1, 'ha': 'right', 'color': 'red'},
                   title='COVID-19 Deaths by Country')

# bar_label_size-柱状图标签文字大小
# tick_label_size-坐标轴标签文字大小
# title_size-标题标签文字大小
bcr.bar_chart_race(df, 'covid19_horiz.gif', bar_label_size=4, tick_label_size=5,
                                 title='COVID-19 Deaths by Country', title_size='smaller')


# shared_fontdict-全局字体属性
bcr.bar_chart_race(df, 'covid19_horiz.gif', title='COVID-19 Deaths by Country',
                                 shared_fontdict={'family': 'Helvetica', 'weight': 'bold',
                                                              'color': 'rebeccapurple'})

# bar_kwargs-条形图属性
bcr.bar_chart_race(df, 'covid19_horiz.gif', bar_kwargs={'alpha': .2, 'ec': 'black', 'lw': 3})


# 设置日期格式，默认为'%Y-%m-%d'
bcr.bar_chart_race(df, 'covid19_horiz.gif', period_fmt='%b %-d, %Y')


# 设置日期标签为数值
bcr.bar_chart_race(df.reset_index(drop=True), 'covid19_horiz.gif', interpolate_period=True,
                                 period_fmt='Index value - {x:.2f}')

# 设置文本位置、数值、大小、颜色等
def summary(values, ranks):
    total_deaths = int(round(values.sum(), -2))
    s = f'Total Deaths - {total_deaths:,.0f}'
    return {'x': .99, 'y': .05, 's': s, 'ha': 'right', 'size': 8}
# 添加文本
bcr.bar_chart_race(df, 'covid19_horiz.gif', period_summary_func=summary)

# 设置垂直条数值，分位数
def func(values, ranks):
    return values.quantile(.9)
# 添加垂直条
bcr.bar_chart_race(df, 'covid19_horiz.gif', perpendicular_bar_func=func)
# 设置柱状图颜色
bcr.bar_chart_race(df, 'covid19_horiz.gif', cmap='accent')
# 去除重复颜色
bcr.bar_chart_race(df, 'covid19_horiz.gif', cmap='accent', filter_column_colors=True)

#中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  #Windows
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB'] #Mac
plt.rcParams['axes.unicode_minus'] = False

import bar_chart_race as bcr
import pandas as pd

# 读取数据
df = pd.read_csv('yuhuanshui.csv', encoding='utf-8', header=0, names=['name', 'number', 'day'])
# 处理数据
df_result = pd.pivot_table(df, values='number', index=['day'], columns=['name'], fill_value=0)
# print(df_result)

# 生成图像
bcr.bar_chart_race(df_result, 'heat.gif', title='我是余欢水演职人员热度排行')

colormaps =
{
    "new_colors": [
        '#ff812c',
        '#ff5a5a',
        '#00c5d2',
        '#a64dff',
        '#4e70f0',
        '#f95dba',
        '#ffce2b'
    ]
}

# 使用自定义的颜色列表
bcr.bar_chart_race(df_result, 'heat.gif', title='我是余欢水演职人员热度排行', cmap='new_colors')
