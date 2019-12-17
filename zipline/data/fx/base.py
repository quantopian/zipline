from interface import Interface


class FXRateReader(Interface):

    def get_rates(self, rate, quote, bases, dts):
        """
        Get rates to convert ``bases`` into ``quote``.

        Parameters
        ----------
        rate : str
            Rate type to load.
        quote : str
            Currency code of the currency into to convert.
        bases : np.array[S3]
            Array of codes of the currencies from which to convert. A single
            currency may appear multiple times.
        dts : pd.DatetimeIndex
            Datetimes for which to load rates.

        Returns
        -------
        rates : np.array
            Array of shape ``(len(dts), len(bases))`` containing fx rates.
            The row at index i corresponds to the dt in dts[i].
            The column at index j corresponds to the base currency in bases[j].
        """
