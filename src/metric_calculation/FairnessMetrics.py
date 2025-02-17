from sklearn.linear_model import SGDClassifier
from typing import Optional
import pandas as pd
import numpy as np

class FairnessMetrics:
    def __init__(self, model: SGDClassifier, y_true: pd.DataFrame, x: pd.DataFrame, model_new: Optional[SGDClassifier] = None,
                 num_columns: Optional[list[str]] = None, cat_columns: Optional[list[str]] = None):
        self.model = model
        self.model_new = model_new
        self.y_true = y_true
        self.x = x
        self.y_pred = self.get_prediction()
        self.num_columns = num_columns
        self.cat_columns = cat_columns

    def false_negative_rate(self, protected_attribute: str):
        protected_attribute_data = self.filter_protected_column(protected_attribute)
        groups = protected_attribute_data.unique()
        false_negative_rates = {}

        for group in groups:
            group_elements = (protected_attribute_data == group).tolist()
            mask = [g and (y == 1) for g, y in zip(group_elements, self.y_true)]

            if sum(mask) > 0:
                false_negatives = sum(p == 0 and m for p, m in zip(self.y_pred, mask))
                false_negative_rates[group] = false_negatives / sum(mask)
            else:
                false_negative_rates[group] = np.nan

        return false_negative_rates

    def false_positive_rate(self, protected_attribute: str):
        protected_attribute_data = self.filter_protected_column(protected_attribute)
        groups = protected_attribute_data.unique()
        false_positive_rates = {}

        for group in groups:
            group_elements = (protected_attribute_data == group).tolist()
            mask = [g and (y == 0) for g, y in zip(group_elements, self.y_true)]  # Changed to actual negatives

            if sum(mask) > 0:
                false_positives = sum(
                    p == 1 and m for p, m in zip(self.y_pred, mask))  # Count positive predictions on negatives
                false_positive_rates[group] = false_positives / sum(mask)
            else:
                false_positive_rates[group] = np.nan

        return false_positive_rates

    def correlation_to_prediction(self):
        y_test_array = np.array(self.y_true)

        correlations = self.x.apply(lambda x: np.corrcoef(x, y_test_array)[0, 1])

        categories = correlations.index.str.split('_').str[0]
        grouped_correlations = pd.DataFrame({'category': categories, 'correlation': correlations.abs()})

        category_correlations = grouped_correlations.groupby('category')['correlation'].mean()

        return category_correlations

    def _calculate_statistical_parity(self, feature, numerical_data, categorical_data):
        y_series = pd.Series(self.y_true) if isinstance(self.y_true, list) else self.y_true

        if feature in numerical_data.columns:
            feature_values = numerical_data[feature]
            bins = pd.qcut(feature_values, q=5, duplicates='drop')
        else:
            cat_cols = [col for col in categorical_data.columns if col.startswith(f"{feature}_")]
            if not cat_cols:
                return None
            feature_values = categorical_data[cat_cols].idxmax(axis=1).str.split('_').str[1]
            bins = feature_values

        grouped = y_series.groupby(bins, observed=False).mean()

        return grouped.max() - grouped.min()

    def get_statistical_parity(self, numerical_data: Optional[pd.DataFrame] = None, categorical_data: Optional[pd.DataFrame] = None):
        parity_results = {}

        for feature in numerical_data.columns:
            parity = self._calculate_statistical_parity(feature, numerical_data, categorical_data)
            parity_results[feature] = parity

        categorical_features = set(col.split('_')[0] for col in categorical_data.columns)
        for feature in categorical_features:
            parity = self._calculate_statistical_parity(feature, numerical_data, categorical_data)
            if parity is not None:
                parity_results[feature] = parity

        return sorted(parity_results.items(), key=lambda x: x[1], reverse=True)

    def get_prediction(self):
        return self.model.predict(self.x)

    def filter_protected_column(self, protected_attribute: str):
        return self.x.filter(regex=f'^{protected_attribute}_').idxmax(axis=1)

