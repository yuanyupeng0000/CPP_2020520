from sqlalchemy import Column, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StockBasic(Base):
    """股票列表
    is_hs	    str	N	是否沪深港通标的，N否 H沪股通 S深股通
    list_status	str	N	上市状态： L上市 D退市 P暂停上市
    exchange	str	N	交易所 SSE上交所 SZSE深交所 HKEX港交所(未上线)
    """
    __tablename__ = 'stock_basic'

    ts_code = Column(String(10), primary_key=True)  # TS代码
    symbol = Column(String(10))         # 股票代码
    name = Column(String(10))           # 股票名称
    area = Column(String(4))            # 所在地域
    industry = Column(String(4))        # 所属行业
    fullname = Column(String(30))       # 股票全称
    enname = Column(String(100))        # 英文全称
    market = Column(String(3))          # 市场类型 （主板/中小板/创业板）
    exchange = Column(String(4))        # 交易所代码
    curr_type = Column(String(3))       # 交易货币
    list_status = Column(String(1))     # 上市状态： L上市 D退市 P暂停上市
    list_date = Column(String(8))       # 上市日期
    delist_date = Column(String(8))     # 退市日期
    is_hs = Column(String(1))           # 是否沪深港通标的，N否 H沪股通 S深股通
