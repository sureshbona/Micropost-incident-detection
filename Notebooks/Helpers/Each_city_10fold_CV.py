# https://nander.cc/writing-custom-cross-validation-methods-grid-search
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class CustomCrossValidation:
    @classmethod
    def split(cls,
              X: pd.DataFrame,
              y: np.ndarray = None,
              groups: np.ndarray = None):
        """Returns to a grouped time series split generator."""
        assert len(X) == len(groups),  (
            "Length of the predictors is not"
            "matching with the groups.")
        # The min max index must be sorted in the range
        for group_idx in range(groups.min(), groups.max() + 1):

            training_group = group_idx

            training_indices = np.where(
                groups == training_group)[0]

            # print("training_indices", len(training_indices))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            for train, test in skf.split(X.iloc[training_indices], y[training_indices]):
                yield train, test
