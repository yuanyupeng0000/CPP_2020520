import pandas as pd
import tushare_data as td

def truncate_update(engine, data, table_name):
    """删除mysql表所有数据，to_sql追加新数据"""
    conn = engine.connect()
    conn.execute('truncate ' + table_name)
    for data_frame in data:
        data_frame.to_sql(table_name, engine, if_exists='append', index=False)


def update_stock_basic(engine, pro, retry_count, pause):
    """更新 股票信息 所有数据"""
    data = td.get_stock_basic(pro, retry_count, pause)
    truncate_update(engine, data, 'stock_basic')

def select_from_db(engine, sql_sentence):
    conn = engine.connect()
    cursor = conn.execute(sql_sentence)
    return cursor
    
