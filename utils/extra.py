from sklearn.feature_extraction.text import TfidfTransformer


class MyTfidfTransformer(TfidfTransformer):
    def fit_transform(self, X, y):
        result = super(MyTfidfTransformer, self).fit_transform(X, y)
        result.sort_indices()
        return result


def clean_one_class_category(Y):
    """ Check for one-class categories, all values zeros or ones. Returns a clean version of Y and categories. """

    # check for columns with only one values. All values equal 0, or 1.
    all_zeros = (Y == 0).all()
    all_ones = (Y == 1).all()
    columns_to_drop = list(Y.columns[all_zeros]) + list(Y.columns[all_ones])

    if columns_to_drop:
        print(f"    REMOVED ONE-CLASS COLUMNS: {columns_to_drop}")
        Y = Y.drop(columns=columns_to_drop)

    return Y, list(Y.columns)
