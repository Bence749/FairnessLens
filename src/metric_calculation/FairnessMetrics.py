from sklearn.linear_model import SGDClassifier
from typing import Optional
import pandas as pd
import numpy as np
import polars as pl
import mercury as mr

class FairnessMetrics:
    def __init__(self, model: SGDClassifier, y_true: pd.DataFrame, x: pd.DataFrame,
                 num_columns: mr.MultiSelect, cat_columns: mr.MultiSelect, protected_columns: mr.MultiSelect,
                 model_new: Optional[SGDClassifier] = None):
        self.model = model
        self.model_new = model_new
        self.y_true = y_true
        self.x = x
        self.y_pred = self.get_prediction()
        self.num_columns: mr.MultiSelect = num_columns
        self.cat_columns: mr.MultiSelect = cat_columns
        self.protected_columns: mr.MultiSelect = protected_columns

    def false_negative_rate(self):
        false_negative_rates = {}

        for feature in self.protected_columns.value:
            protected_attribute_data = self.filter_protected_column(feature)
            groups = protected_attribute_data.keys()
            false_negative_rates_inner = {}

            for group in groups:
                mask = [g and (y == 1) for g, y in
                        zip(protected_attribute_data[group], self.y_true)]  # Changed to actual negatives

                if sum(mask) > 0:
                    false_negatives = sum(
                        p == 0 and m for p, m in zip(self.y_pred, mask))  # Count negative predictions on negatives
                    false_negative_rates_inner[group] = false_negatives / sum(mask)
                else:
                    false_negative_rates_inner[group] = np.nan

            false_negative_rates[feature] = false_negative_rates_inner

        return false_negative_rates

    def false_positive_rate(self):
        false_positive_rates = {}
        for feature in self.protected_columns.value:
            protected_attribute_data = self.filter_protected_column(feature)
            groups = protected_attribute_data.keys()
            false_positive_rates_inner = {}

            for group in groups:
                mask = [g and (y == 0) for g, y in zip(protected_attribute_data[group], self.y_true)]  # Changed to actual negatives

                if sum(mask) > 0:
                    false_positives = sum(
                        p == 1 and m for p, m in zip(self.y_pred, mask))  # Count positive predictions on negatives
                    false_positive_rates_inner[group] = false_positives / sum(mask)
                else:
                    false_positive_rates_inner[group] = np.nan

            false_positive_rates[feature] = false_positive_rates_inner

        return false_positive_rates

    def correlation_to_prediction(self):
        y_true_array = np.array(self.y_true).flatten()
        correlations = {}

        for feature in self.protected_columns.value:
            # Handle numerical features
            if feature in self.num_columns.value:
                col_data = self.x[feature]
                if len(col_data) < 2:
                    correlations[feature] = np.nan
                    continue
                with np.errstate(invalid='ignore'):
                    corr = np.corrcoef(col_data, y_true_array)[0, 1]
                correlations[feature] = np.abs(corr) if not np.isnan(corr) else corr

            # Handle categorical features
            elif feature in self.cat_columns.value:
                pattern = f'^{feature}_'
                cat_cols = self.x.columns[self.x.columns.str.match(pattern)]

                if not cat_cols.empty:
                    cat_corrs = []
                    for col in cat_cols:
                        col_data = self.x[col]
                        if len(col_data) < 2:
                            continue
                        with np.errstate(invalid='ignore'):
                            corr = np.corrcoef(col_data, y_true_array)[0, 1]
                        if not np.isnan(corr):
                            cat_corrs.append(np.abs(corr))

                    correlations[feature] = np.mean(cat_corrs) if cat_corrs else np.nan
                else:
                    correlations[feature] = np.nan

        return correlations

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

    def get_statistical_parity(self, numerical_data, categorical_data):
        parity_results = {}

        for feature in self.protected_columns.value:
            if feature in numerical_data.columns:
                parity = self._calculate_statistical_parity(feature, numerical_data, categorical_data)
                parity_results[feature] = parity

            if feature in self.cat_columns.value:
                parity = self._calculate_statistical_parity(feature, numerical_data, categorical_data)
                if parity is not None:
                    parity_results[feature] = parity

        return sorted(parity_results.items(), key=lambda x: x[1], reverse=True)

    def get_prediction(self):
        return self.model.predict(self.x)

    def filter_protected_column(self, protected_attribute: str):
        # Handle numerical features
        if protected_attribute in self.num_columns.value:
            return self.x[[protected_attribute]]

        # Handle categorical features
        return self.x.filter(regex=f'^{protected_attribute}_')

    def get_polars_metrics(self) -> pl.DataFrame:
        output = pl.DataFrame()
        output = output.with_columns(
            feature=self.protected_columns.value,
            correlation=pl.lit(list(self.correlation_to_prediction().values())),
            fnr=pl.lit(self.false_negative_rate()),
            fpr=pl.lit(self.false_positive_rate()),
            # parity=pl.lit(self.get_statistical_parity())
        )
        return output
