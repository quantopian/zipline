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
        rates : pd.DataFrame
            DataFrame indexed by (dts, bases) containing exchange rates mapping
            from base -> quote currency.
        """
