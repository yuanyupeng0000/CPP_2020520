import time
import pandas as pd


def get_stock_basic(pro, retry_count=3, pause=2):
    """股票列表 数据"""
    frame = []
    for status in ['L', 'D', 'P']:
        for _ in range(retry_count):
            try:
                df = pro.stock_basic(exchange='', list_status=status,
                                     fields='ts_code,symbol,name,area,industry,fullname,enname,market, \
                                    exchange,curr_type,list_status,list_date,delist_date,is_hs')
                print(df)
            except:
                time.sleep(pause)
            else:
                frame.append(df)
                break

    return frame

def analize_one(ts_code):
    score = 0

    return score
