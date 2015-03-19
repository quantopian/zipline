

from sqlalchemy import Column, Float, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TestOrm(Base):

    __tablename__ = 'TEST_TABLE'

    index = Column(TIMESTAMP,  primary_key=True)
    price = Column(Float)
    sid = Column(Integer, primary_key=True)

