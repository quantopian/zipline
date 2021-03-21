import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)

backtrace = Path('backtrace.txt')
errors = backtrace.read_text().split('======================================================================')[1:]
results = []
for error in errors:
    txt = error.strip().split('\n')
    results.append([txt[0].strip(), txt[-1].strip('. ')])

results = pd.DataFrame(results, columns=['case', 'error'])
print(results.head())
print(results.error.nunique())
print(results.error.value_counts().head(30))
