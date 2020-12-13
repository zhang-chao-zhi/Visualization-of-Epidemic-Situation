#!/usr/bin/env python
# -*- coding:utf-8 -*-

from atrader import *


def init(context: Context):
    # 设置初始资金为100万，期货交易手续费为交易所的1.1倍，股票交易手续费为万分之2，保证金率为交易所1.1倍
    set_backtest(initial_cash=1000000, future_cost_fee=1.1, stock_cost_fee=2, margin_rate=1.1)


def on_data(context: Context):
    pass


if __name__ == '__main__':
    # 设置回测区间为2018-01-01至2018-06-30
    # 设置刷新频率为15min
    # 设置策略需要的标的为螺纹钢主力连续合约
    run_backtest(strategy_name='test', file_path='.', target_list=['SHFE.RB0000'], frequency='min', fre_num=15,
                 begin_date='2020-01-01', end_date='2020-11-09')
    target_list = ['CZCE.FG000', 'SHFE.RB0000']  # 设置回测标的
    frequency = 'min'  # 设置刷新频率
    fre_num = 1  # 设置刷新频率
    begin_date = '2020-01-01'  # 设置回测初始时间
    end_date = '2020-11-09'  # 设置回测结束时间
    fq = 1  # 设置复权方式
    # 将对应的参数放进策略回测入口函数
    strategy_id = run_backtest('海龟', '.', target_list=target_list, frequency=frequency, fre_num=fre_num,
                               begin_date=begin_date, end_date=end_date, fq=fq)