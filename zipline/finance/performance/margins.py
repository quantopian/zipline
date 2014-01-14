#
# Copyright 2013 Quantopian, Inc.
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

'''
Margin Requirements according to FINRA rules.

Rules and interpretations can be found at
http://www.finra.org/web/groups/industry/@ip/@reg/@rules/documents/industry/p122203.pdf
'''

def calculate_margin_requirement(position):
    ''' Calculates the maintenance margin required by FINRA. '''
    amount = position.amount
    last_sale_price = position.last_sale_price
    if amount >= 0:
        req = .25 * amount * last_sale_price
    else:
        if last_sale_price < 5:  
            req = max(2.5 * amount, abs(amount * last_sale_price))  
        else:  
            req = max(5 * amount, abs(0.3 * amount * last_sale_price))
    position.margin_requirement = req
    return position.margin_requirement


class Margins(object):
    
    def __init__(self, portfolio=None, day_trader=False, 
                 leverage=None, intraday_calls=True, exemptions=None):
        self.initial_margin = 25000.0 if day_trader else 2000.0
        self.intraday_calls = intraday_calls
        self.exemptions = exemptions        
        self.position_margins = dict()
        for pos in portfolio.positions:
            self.position_margins[pos] =\
                calculate_margin_requirement(portfolio.positions[pos])
        self.requirement = sum(
            [self.position_margins[i] for i in self.position_margins]
        )
    
    def __repr__(self):
        template = "Total Requirement: {req}, position_margins: {position_margins}"
        return template.format(
            req = self.requirement,
            position_margins = self.position_margins
        )    
    def __getitem__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            return self.position_margins[item]
