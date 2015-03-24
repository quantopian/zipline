#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Object relational mapping used for unittests.
"""
from sqlalchemy import Column, Float, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TestOrm(Base):

    __tablename__ = 'TEST_TABLE'

    index = Column(TIMESTAMP,  primary_key=True)
    price = Column(Float)
    sid = Column(Integer, primary_key=True)
