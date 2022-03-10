import numpy as np
import pandas

from behavior import BASE_TARGET_COLS
from behavior.modeling.featurewiz import featurewiz as FW

# To prevent having any actual zeros in the training/testing data which can yield
# Inf, we bias each observation by a small epsilon reflected below. The epsilon is
# added so our X's and Y's no longer includes 0.0.
BIAS_EPSILON = 1e-6


def derive_input_features(train, test=None, targets=None, config=None):
    """
    Derives input features for a dataframe using featurewiz.
    Dataframe assumes that all BASE_TARGET_COLS exist.

    Parameters
    ----------
    train : pandas.DataFrame
        DataFrame to derive input features from.
    test : pandas.DataFrame
        DataFrame to use for testing the input features. If this is not specified,
        featurewiz will split [train] into a test set.
    targets : list[str]
        List of columns from train to optimize for. All other Y's from train
        are dropped before train is passed to featurewiz.
    config : dict[str, Any]
        Configuration parameters.

    Returns
    -------
    metadata : list[str]
        List of strings that describe how to transform the input data into input features
        expected by the model.
    """
    assert len(targets) > 0, "No targets specified for derive_input_features()"

    for column in train.columns:
        # These checks ensure that the transformation done by featurewiz
        # (featurewiz/databunch.py: gen_numeric_interaction_features())
        # can be reversed successfully. Particularly, we require that the
        # column name cannot contain any of the following keywords.
        blocked = ["_div_by_", "_mult_by_", "_minus_", "_plus_", "_squared"]
        for block in blocked:
            assert block not in column, f"Found blocked {block} inside column name {column}."

    # Drop all target columns that we are not trying to optimize for.
    # For instance, if targets=["elapsed_us"], then we drop all other Y's.
    # This prevents featurewiz from trying to use other Y's as X's.
    drop_targets = set(BASE_TARGET_COLS) - set(targets)
    train_input = train.copy(deep=True)
    test_input = test.copy(deep=True) if test is not None else None
    if len(drop_targets) > 0:
        train_input.drop(drop_targets, axis=1, inplace=True)
        if test_input is not None:
            test_input.drop(drop_targets, axis=1, inplace=True)

    # Add the EPSILON to remove zeros from the dataset.
    train_input = train_input + BIAS_EPSILON
    if test_input is not None:
        test_input = test_input + BIAS_EPSILON

    # Invoke featurewiz to find the best features to use.
    features, _ = FW.featurewiz(
        train_input,
        targets,
        corr_limit=config["corr_limit"] if config and "corr_limit" in config else 0.70,
        verbose=config["verbose"] if config and "verbose" in config else 0,
        test_data=test_input,
        # Target Encoding and Group By aren't useful since we don't actually
        # have any true "categorical" features.
        feature_engg="interactions",
        category_encoders="",
        dask_xgboost_flag=False,
        nrows=None,
    )

    if len(features.columns) - len(targets) == 0:
        # In this case, featurewiz thinks all the features are irrelevant.
        # For this, we add a fake "bias" attribute.
        return ["bias"]

    return list(features.columns)[0 : -len(targets)]


def extract_all_features(df):
    """
    Derives input features for a dataframe as all non-target columns.
    Dataframe assumes that all BASE_TARGET_COLS exist.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to derive input features from.

    Returns
    -------
    metadata : list[str]
        List of strings that describe how to transform the input data into input features
        expected by the model.
    """
    input_columns = set(df.columns) - set(BASE_TARGET_COLS)
    return list(input_columns)


def extract_input_features(df, metadata):
    """
    Given a base dataframe (e.g., of the form extracted by TScout/Hutch),
    and the metadata returned by derive_input_features(), this function
    extracts/reconstructs the input features for the model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of input X's to convert. All features in df must be numeric.
    metadata : list[str]
        List of strings describing how to transform df into the model's features.

    Returns
    -------
    pandas.DataFrame
        Input dataframe transformed into the model's input features.
    """

    # To prevent some transformations from yielding Inf, we strip all zeros
    # from the input data by adding a small epsilon.
    df = df + BIAS_EPSILON
    output_columns = []
    for column in metadata:

        # These checks reverse the transformation done by featurewiz.
        # featurewiz/databunch.py: gen_numeric_interaction_features()

        # ops is a map from [id] -> [func]. Splitting the column with
        # [id] produces a list of input columns from df that are passed
        # as inputs to [func].
        ops = {
            "_div_by_": lambda x, y: x * 1.0 / y,
            "_mult_by_": lambda x, y: x * y,
            "_minus_": lambda x, y: x - y,
            "_plus_": lambda x, y: x + y,
            "_squared": lambda x: x.pow(2),
        }

        found = False
        for (key, func) in ops.items():
            if key in column:
                found = True

                # Split the column using the key.
                components = column.split(key)
                # Get columns from df based on the split components.
                columns = [df[component] for component in components if component != ""]
                # Compute the derived column
                output = func(*columns)
                output.name = column
                output_columns.append(output)

        if not found:
            if column == "bias":
                # Create the fake bias column.
                ones = np.ones(df.shape[0])
                bias = pandas.Series(data=ones, name=column, copy=False)
                output_columns.append(bias)
            else:
                # In the case that an op is not found, we use the column directly.
                output_columns.append(df[[column]])

    output = pandas.concat(output_columns, axis=1)
    return output
