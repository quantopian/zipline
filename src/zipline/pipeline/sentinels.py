from zipline.utils.sentinel import sentinel


NotSpecified = sentinel(
    "NotSpecified",
    "Singleton sentinel value used for Term defaults.",
)

NotSpecifiedType = type(NotSpecified)
