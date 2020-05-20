import pandas as pd
import tushare as ts
import pymysql
pymysql.install_as_MySQLdb()
from sqlalchemy import create_engine

from mysql_tables_structure import Base
import mysql_functions as mf

# Create db engine
engine = create_engine('mysql://root:305128@127.0.0.1/stock?charset=utf8mb4') 
conn = engine.connect()

# Create all table structure
Base.metadata.create_all(engine)

# Connect tushare
ts.set_token('2870c8f839dff2c02cff693197ea0daff9f1320642cfa65fdc104df3')
pro = ts.pro_api()

# Update stock list
mf.update_stock_basic(engine, pro, 3, 2)

# Get all stock code
sql_sentence = 'select ts_code from stock.stock_basic'
sqlalchemy_result_proxy = mf.select_from_db(engine, sql_sentence)
#dict((zip(sqlalchemy_result_proxy.keys(), sqlalchemy_result_proxy)))
codes_list = [i[0] for i in list(sqlalchemy_result_proxy)]
print("len:{0} \n{1}".format(len(codes_list), codes_list))
