

from sqlalchemy import Column, Float, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TestOrm(Base):

    __tablename__ = 'TEST_TABLE'

    index = Column(TIMESTAMP, primary_key=True)
    price = Column(Float, key="0")

    def __repr__(self):
        return ('%19s' % (
                self.index
                ))