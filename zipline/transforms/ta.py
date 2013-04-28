import talib
from zipline.transforms import BatchTransform


def make_transform(talib_fn):
    """
    A factory for BatchTransforms based on TALIB abstract functions.
    """
    class TALibTransform(BatchTransform):
        def __init__(self, sid, refresh_period=0, **kwargs):

            self.talib_fn = talib_fn

            # get default talib parameters
            self.talib_parameters = talib_fn.get_parameters()

            # update new parameters from kwargs
            for k, v in kwargs.iteritems():
                if k in self.talib_parameters:
                    self.talib_parameters[k] = v

            def zipline_wrapper(data):
                # Set the parameters at each iteration in case the same
                # abstract talib function is being used in another
                # BatchTransform with different parameters.
                # FIXME -- this might not be necessary if the abstract
                # functions can be copied into separate objects.
                self.talib_fn.set_parameters(self.talib_parameters)

                # convert zipline dataframe to talib data_dict
                data_dict = dict()
                #TODO handle missing data
                for key in ['open', 'high', 'low', 'volume']:
                    if key in data:
                        data_dict[key] = data[key].values[:, 0]
                if 'close' in data:
                    data_dict['close'] = data['close'].values[:, 0]
                else:
                    data_dict['close'] = data['price'].values[:, 0]

                # call talib
                result = self.talib_fn(data_dict)

                # keep only the most recent result
                if isinstance(result, (list, tuple)):
                    return tuple([r[-1] for r in result])
                else:
                    return result[-1]

            super(TALibTransform, self).__init__(
                func=zipline_wrapper,
                sids=sid,
                refresh_period=refresh_period,
                window_length=max(1, self.talib_fn.lookback + 1))

    # bind a class docstring to reveal parameters
    TALibTransform.__doc__ = getattr(talib, talib_fn.info['name']).__doc__

    #return class
    return TALibTransform


ACOS = make_transform(talib.abstract.ACOS)
AD = make_transform(talib.abstract.AD)
ADD = make_transform(talib.abstract.ADD)
ADOSC = make_transform(talib.abstract.ADOSC)
ADX = make_transform(talib.abstract.ADX)
ADXR = make_transform(talib.abstract.ADXR)
APO = make_transform(talib.abstract.APO)
AROON = make_transform(talib.abstract.AROON)
AROONOSC = make_transform(talib.abstract.AROONOSC)
ASIN = make_transform(talib.abstract.ASIN)
ATAN = make_transform(talib.abstract.ATAN)
ATR = make_transform(talib.abstract.ATR)
AVGPRICE = make_transform(talib.abstract.AVGPRICE)
BBANDS = make_transform(talib.abstract.BBANDS)
BETA = make_transform(talib.abstract.BETA)
BOP = make_transform(talib.abstract.BOP)
CCI = make_transform(talib.abstract.CCI)
CDL2CROWS = make_transform(talib.abstract.CDL2CROWS)
CDL3BLACKCROWS = make_transform(talib.abstract.CDL3BLACKCROWS)
CDL3INSIDE = make_transform(talib.abstract.CDL3INSIDE)
CDL3LINESTRIKE = make_transform(talib.abstract.CDL3LINESTRIKE)
CDL3OUTSIDE = make_transform(talib.abstract.CDL3OUTSIDE)
CDL3STARSINSOUTH = make_transform(talib.abstract.CDL3STARSINSOUTH)
CDL3WHITESOLDIERS = make_transform(talib.abstract.CDL3WHITESOLDIERS)
CDLABANDONEDBABY = make_transform(talib.abstract.CDLABANDONEDBABY)
CDLADVANCEBLOCK = make_transform(talib.abstract.CDLADVANCEBLOCK)
CDLBELTHOLD = make_transform(talib.abstract.CDLBELTHOLD)
CDLBREAKAWAY = make_transform(talib.abstract.CDLBREAKAWAY)
CDLCLOSINGMARUBOZU = make_transform(talib.abstract.CDLCLOSINGMARUBOZU)
CDLCONCEALBABYSWALL = make_transform(talib.abstract.CDLCONCEALBABYSWALL)
CDLCOUNTERATTACK = make_transform(talib.abstract.CDLCOUNTERATTACK)
CDLDARKCLOUDCOVER = make_transform(talib.abstract.CDLDARKCLOUDCOVER)
CDLDOJI = make_transform(talib.abstract.CDLDOJI)
CDLDOJISTAR = make_transform(talib.abstract.CDLDOJISTAR)
CDLDRAGONFLYDOJI = make_transform(talib.abstract.CDLDRAGONFLYDOJI)
CDLENGULFING = make_transform(talib.abstract.CDLENGULFING)
CDLEVENINGDOJISTAR = make_transform(talib.abstract.CDLEVENINGDOJISTAR)
CDLEVENINGSTAR = make_transform(talib.abstract.CDLEVENINGSTAR)
CDLGAPSIDESIDEWHITE = make_transform(talib.abstract.CDLGAPSIDESIDEWHITE)
CDLGRAVESTONEDOJI = make_transform(talib.abstract.CDLGRAVESTONEDOJI)
CDLHAMMER = make_transform(talib.abstract.CDLHAMMER)
CDLHANGINGMAN = make_transform(talib.abstract.CDLHANGINGMAN)
CDLHARAMI = make_transform(talib.abstract.CDLHARAMI)
CDLHARAMICROSS = make_transform(talib.abstract.CDLHARAMICROSS)
CDLHIGHWAVE = make_transform(talib.abstract.CDLHIGHWAVE)
CDLHIKKAKE = make_transform(talib.abstract.CDLHIKKAKE)
CDLHIKKAKEMOD = make_transform(talib.abstract.CDLHIKKAKEMOD)
CDLHOMINGPIGEON = make_transform(talib.abstract.CDLHOMINGPIGEON)
CDLIDENTICAL3CROWS = make_transform(talib.abstract.CDLIDENTICAL3CROWS)
CDLINNECK = make_transform(talib.abstract.CDLINNECK)
CDLINVERTEDHAMMER = make_transform(talib.abstract.CDLINVERTEDHAMMER)
CDLKICKING = make_transform(talib.abstract.CDLKICKING)
CDLKICKINGBYLENGTH = make_transform(talib.abstract.CDLKICKINGBYLENGTH)
CDLLADDERBOTTOM = make_transform(talib.abstract.CDLLADDERBOTTOM)
CDLLONGLEGGEDDOJI = make_transform(talib.abstract.CDLLONGLEGGEDDOJI)
CDLLONGLINE = make_transform(talib.abstract.CDLLONGLINE)
CDLMARUBOZU = make_transform(talib.abstract.CDLMARUBOZU)
CDLMATCHINGLOW = make_transform(talib.abstract.CDLMATCHINGLOW)
CDLMATHOLD = make_transform(talib.abstract.CDLMATHOLD)
CDLMORNINGDOJISTAR = make_transform(talib.abstract.CDLMORNINGDOJISTAR)
CDLMORNINGSTAR = make_transform(talib.abstract.CDLMORNINGSTAR)
CDLONNECK = make_transform(talib.abstract.CDLONNECK)
CDLPIERCING = make_transform(talib.abstract.CDLPIERCING)
CDLRICKSHAWMAN = make_transform(talib.abstract.CDLRICKSHAWMAN)
CDLRISEFALL3METHODS = make_transform(talib.abstract.CDLRISEFALL3METHODS)
CDLSEPARATINGLINES = make_transform(talib.abstract.CDLSEPARATINGLINES)
CDLSHOOTINGSTAR = make_transform(talib.abstract.CDLSHOOTINGSTAR)
CDLSHORTLINE = make_transform(talib.abstract.CDLSHORTLINE)
CDLSPINNINGTOP = make_transform(talib.abstract.CDLSPINNINGTOP)
CDLSTALLEDPATTERN = make_transform(talib.abstract.CDLSTALLEDPATTERN)
CDLSTICKSANDWICH = make_transform(talib.abstract.CDLSTICKSANDWICH)
CDLTAKURI = make_transform(talib.abstract.CDLTAKURI)
CDLTASUKIGAP = make_transform(talib.abstract.CDLTASUKIGAP)
CDLTHRUSTING = make_transform(talib.abstract.CDLTHRUSTING)
CDLTRISTAR = make_transform(talib.abstract.CDLTRISTAR)
CDLUNIQUE3RIVER = make_transform(talib.abstract.CDLUNIQUE3RIVER)
CDLUPSIDEGAP2CROWS = make_transform(talib.abstract.CDLUPSIDEGAP2CROWS)
CDLXSIDEGAP3METHODS = make_transform(talib.abstract.CDLXSIDEGAP3METHODS)
CEIL = make_transform(talib.abstract.CEIL)
CMO = make_transform(talib.abstract.CMO)
CORREL = make_transform(talib.abstract.CORREL)
COS = make_transform(talib.abstract.COS)
COSH = make_transform(talib.abstract.COSH)
DEMA = make_transform(talib.abstract.DEMA)
DIV = make_transform(talib.abstract.DIV)
DX = make_transform(talib.abstract.DX)
EMA = make_transform(talib.abstract.EMA)
EXP = make_transform(talib.abstract.EXP)
FLOOR = make_transform(talib.abstract.FLOOR)
HT_DCPERIOD = make_transform(talib.abstract.HT_DCPERIOD)
HT_DCPHASE = make_transform(talib.abstract.HT_DCPHASE)
HT_PHASOR = make_transform(talib.abstract.HT_PHASOR)
HT_SINE = make_transform(talib.abstract.HT_SINE)
HT_TRENDLINE = make_transform(talib.abstract.HT_TRENDLINE)
HT_TRENDMODE = make_transform(talib.abstract.HT_TRENDMODE)
KAMA = make_transform(talib.abstract.KAMA)
LINEARREG = make_transform(talib.abstract.LINEARREG)
LINEARREG_ANGLE = make_transform(talib.abstract.LINEARREG_ANGLE)
LINEARREG_INTERCEPT = make_transform(talib.abstract.LINEARREG_INTERCEPT)
LINEARREG_SLOPE = make_transform(talib.abstract.LINEARREG_SLOPE)
LN = make_transform(talib.abstract.LN)
LOG10 = make_transform(talib.abstract.LOG10)
MA = make_transform(talib.abstract.MA)
MACD = make_transform(talib.abstract.MACD)
MACDEXT = make_transform(talib.abstract.MACDEXT)
MACDFIX = make_transform(talib.abstract.MACDFIX)
MAMA = make_transform(talib.abstract.MAMA)
MAVP = make_transform(talib.abstract.MAVP)
MAX = make_transform(talib.abstract.MAX)
MAXINDEX = make_transform(talib.abstract.MAXINDEX)
MEDPRICE = make_transform(talib.abstract.MEDPRICE)
MFI = make_transform(talib.abstract.MFI)
MIDPOINT = make_transform(talib.abstract.MIDPOINT)
MIDPRICE = make_transform(talib.abstract.MIDPRICE)
MIN = make_transform(talib.abstract.MIN)
MININDEX = make_transform(talib.abstract.MININDEX)
MINMAX = make_transform(talib.abstract.MINMAX)
MINMAXINDEX = make_transform(talib.abstract.MINMAXINDEX)
MINUS_DI = make_transform(talib.abstract.MINUS_DI)
MINUS_DM = make_transform(talib.abstract.MINUS_DM)
MOM = make_transform(talib.abstract.MOM)
MULT = make_transform(talib.abstract.MULT)
NATR = make_transform(talib.abstract.NATR)
OBV = make_transform(talib.abstract.OBV)
PLUS_DI = make_transform(talib.abstract.PLUS_DI)
PLUS_DM = make_transform(talib.abstract.PLUS_DM)
PPO = make_transform(talib.abstract.PPO)
ROC = make_transform(talib.abstract.ROC)
ROCP = make_transform(talib.abstract.ROCP)
ROCR = make_transform(talib.abstract.ROCR)
ROCR100 = make_transform(talib.abstract.ROCR100)
RSI = make_transform(talib.abstract.RSI)
SAR = make_transform(talib.abstract.SAR)
SAREXT = make_transform(talib.abstract.SAREXT)
SIN = make_transform(talib.abstract.SIN)
SINH = make_transform(talib.abstract.SINH)
SMA = make_transform(talib.abstract.SMA)
SQRT = make_transform(talib.abstract.SQRT)
STDDEV = make_transform(talib.abstract.STDDEV)
STOCH = make_transform(talib.abstract.STOCH)
STOCHF = make_transform(talib.abstract.STOCHF)
STOCHRSI = make_transform(talib.abstract.STOCHRSI)
SUB = make_transform(talib.abstract.SUB)
SUM = make_transform(talib.abstract.SUM)
T3 = make_transform(talib.abstract.T3)
TAN = make_transform(talib.abstract.TAN)
TANH = make_transform(talib.abstract.TANH)
TEMA = make_transform(talib.abstract.TEMA)
TRANGE = make_transform(talib.abstract.TRANGE)
TRIMA = make_transform(talib.abstract.TRIMA)
TRIX = make_transform(talib.abstract.TRIX)
TSF = make_transform(talib.abstract.TSF)
TYPPRICE = make_transform(talib.abstract.TYPPRICE)
ULTOSC = make_transform(talib.abstract.ULTOSC)
VAR = make_transform(talib.abstract.VAR)
WCLPRICE = make_transform(talib.abstract.WCLPRICE)
WILLR = make_transform(talib.abstract.WILLR)
WMA = make_transform(talib.abstract.WMA)
