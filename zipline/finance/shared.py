
from abc import ABCMeta
from itertools import chain


class FinancialModelMeta(ABCMeta):
    """
    This metaclass allows users to create custom slippage and commission models
    that support both equities and futures by subclassing the appropriate
    individualized classes.

    Take for example the following custom model, which subclasses both
    EquitySlippageModel and FutureSlippageModel together:

        class MyCustomSlippage(EquitySlippageModel, FutureSlippageModel):
            def process_order(self, data, order):
                ...

    Ordinarily the first parent class in the MRO ('EquitySlippageModel' in this
    example) would override the 'allowed_asset_types' class attribute to only
    include equities. However, this is probably not the expected behavior for a
    reasonable user, so the metaclass will handle this specific case by
    manually allowing both equities and futures for the class being created.
    """

    def __new__(mcls, name, bases, dict_):
        if 'allowed_asset_types' not in dict_:
            allowed_asset_types = tuple(
                chain.from_iterable(
                    marker.allowed_asset_types
                    for marker in bases
                    if isinstance(marker, AllowedAssetMarker)
                )
            )
            if allowed_asset_types:
                dict_['allowed_asset_types'] = allowed_asset_types

        return super(FinancialModelMeta, mcls).__new__(
            mcls, name, bases, dict_,
        )


class AllowedAssetMarker(FinancialModelMeta):
    pass
