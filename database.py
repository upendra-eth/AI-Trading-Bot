import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Opportunity(Base):
    __tablename__ = 'opportunities'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(50), nullable=False)
    xgb_signal = Column(Float, nullable=True)     # Range e.g. -1.0 to 1.0 or confidence
    lstm_signal = Column(Float, nullable=True)
    finbert_signal = Column(Float, nullable=True)
    final_signal = Column(String(10), nullable=False) # BUY, SELL, HOLD
    executed = Column(Boolean, default=False)

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=False)
    pnl = Column(Float, default=0.0)
    status = Column(String(20), default='OPEN') # OPEN, CLOSED

class Portfolio(Base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True, autoincrement=True)
    balance = Column(Float, nullable=False, default=100000.0) # Paper trading starting balance

def init_db(db_path='sqlite:///trading.db'):
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Initialize portfolio if not exists
    if session.query(Portfolio).count() == 0:
        session.add(Portfolio(balance=100000.0))
        session.commit()
        
    return Session

if __name__ == '__main__':
    # When run directly, initialize database in current directory
    init_db()
    print("Database trading.db initialized successfully.")
