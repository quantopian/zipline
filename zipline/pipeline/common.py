"""
Common constants for Pipeline.
"""
AD_FIELD_NAME = 'asof_date'
ANNOUNCEMENT_FIELD_NAME = 'announcement_date'
CASH_FIELD_NAME = 'cash'
DAYS_SINCE_PREV = 'days_since_prev'
DAYS_TO_NEXT = 'days_to_next'
FISCAL_QUARTER_FIELD_NAME = 'fiscal_quarter'
FISCAL_YEAR_FIELD_NAME = 'fiscal_year'
NEXT_ANNOUNCEMENT = 'next_announcement'
PREVIOUS_AMOUNT = 'previous_amount'
PREVIOUS_ANNOUNCEMENT = 'previous_announcement'

EVENT_DATE_FIELD_NAME = 'event_date'
SID_FIELD_NAME = 'sid'
TS_FIELD_NAME = 'timestamp'

DATE_INDEX_NAME = 'dates'
SID_INDEX_NAME = 'sid'
PIPELINE_INDEX_NAMES = (DATE_INDEX_NAME, SID_INDEX_NAME)
