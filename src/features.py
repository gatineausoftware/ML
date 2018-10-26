import json


def get_postalcode():
    """Create lat/lon postalcode features from a source postalcode.
    Args:
        df: Dataframe being changed by transformation.
        name: Base name of new features.
        source: Source feature being transformed in dataframe.
    Returns:
        None.
    """
    postalcode_info = json.load(open('postalcode_info.json'))

    def get_lat(x):
        return postalcode_info[str(x)]['lat']

    def get_long(x):
        return postalcode_info[str(x)]['lat']

    def postal_code(df, name, source):
        df[name + '_lat'] = df[source].apply(
            lambda x: postalcode_info[str(x)]['lat'], meta=('x', float))
        df[name + '_lon'] = df[source].apply(
            lambda x: postalcode_info[str(x)]['lon'], meta=('x', float))

    return postal_code






