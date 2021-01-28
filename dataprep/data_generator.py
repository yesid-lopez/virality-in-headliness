import tensorflow as tf
import numpy as np


@tf.autograph.experimental.do_not_convert
def single_data_generator(df, packages_ids, one_hot=False, classification=True, regression=False,
                          sorted=False, cased=False, features=None, enforced=False, isolate=None):
    """
    Yields a single training sample according to classification or regression task.

    Arguments:
      df(pd.DataFrame): DataFrame from which headlines will be retrieved
      packages_ids(list or numpy array): List of packages ids where pairs of
                    headlines will be taken randomly

      one_hot(bool): whether to one hot encode class labels. Only valid if
               classification = True, otherwise a ValueError will be raised

      classification(bool): bool, whether to yield labels as a binary classification task.
                      1 ([0, 1] if one_hot=True) if first headline had more clicks than
                      the second headline, 0 ([1, 0] if one_hot=True) otherwise.

      regression(bool): bool, whether to yield targets as a regression task.
                  The regression is calculated as follows:

                  abs(num_clicks_in_headline_1 - num_clicks_in_headline_2) /
                            maximum_number_of_clicks_in_package

          regression and classification are not mutually exclusive, if both are set
          to True, then a tuple with

      sorted(bool): If True, yielded headlines will be sorted in descending order by number of clicks.
              Otherwise, yielded headlines will not have any particular order

      cased(bool): If True, yielded headlines will not be lowercased.
             Otherwise, all headlines will be lowercased,

      features(list): default to None. List of names of the variables to include as features in df;
                yielded elements will include a vector with the values of the features specified for each headline.

                Concretely:

                  (headline0, *features0, headline1, *features1, regression, classification) is yielded

                Otherwise, if features=None (default):

                  (headline1, headline2, regression, classification | regression) is yielded

      enforced: If True, then the same pair of headlines are yielded twice,
                but the second pair is swapped  with labels and features (if valid)
                changed accordingly.

      isolate: Defaults to None. If different than None, then yields a string
               which only has distinct words between two headlines and the other
               tokens are padded with a isolate string

    """
    if not classification and not regression:
        raise ValueError("Output must be either classification or regression (not exclusive)")
    if one_hot and not classification:
        raise ValueError("One hot encoding is only allowed in classification mode")
    if isolate is not None and type(isolate) != str:
        raise ValueError("Isolate can only be a string")

    while True:
        idx = np.squeeze(np.random.choice(len(packages_ids), 1, replace=False))
        chosen_package_id = str(packages_ids[idx])
        package = df[df["clickability_test_id"] == chosen_package_id]
        random_headline_idxs = np.random.choice(len(package), 2, replace=False)
        y_raw = package.clicks.values[random_headline_idxs]

        # Skip if both headlines have the same number of clicks
        if y_raw[0] == y_raw[1]:
            continue

        # Get both headlines
        if not cased:
            h1, h2 = package["headline_lowercase"].values[random_headline_idxs].tolist()
        else:
            h1, h2 = package["headline"].values[random_headline_idxs].tolist()

        # Skip if both headlines are the same
        if h1 == h2:
            continue

        label = int(y_raw[0] > y_raw[1])

        y_classification = -1
        y_regression = -1

        if classification:
            y_classification = np.eye(2)[label] if one_hot else label

        if regression:
            y_regression = np.abs(np.subtract(*y_raw) / package.clicks.max())

        if sorted:
            headlines = (h1, h2) if label else (h2, h1)
        else:
            headlines = (h1, h2)
        if not enforced:
            inputs = (headlines[0], headlines[1], y_regression, y_classification)
            if features is not None:
                features_h0, features_h1 = package[features].values[random_headline_idxs]
                inputs = (headlines[0], *features_h0, headlines[1], *features_h1, y_regression, y_classification)
            yield inputs
        else:
            swapped_yclassification = int(not y_classification if y_classification != -1 else y_classification)
            inputs1 = (headlines[0], headlines[1], y_regression, y_classification)
            inputs2 = (headlines[1], headlines[0], y_regression, swapped_yclassification)
            if features is not None:
                features_h0, features_h1 = package[features].values[random_headline_idxs]
                inputs1 = (headlines[0], *features_h0, headlines[1], *features_h1, y_regression, y_classification)
                inputs2 = (headlines[1], *features_h1, headlines[0], *features_h0, y_regression,
                           swapped_yclassification)  # Swapped
            for inputs in [inputs1, inputs2]:
                yield inputs
