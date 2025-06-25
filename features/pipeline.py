from sklearn.pipeline import FeatureUnion, Pipeline
from .factors import MomemntumTransformer, ZScoreTransformer

def create_feature_union() -> FeatureUnion:
    """
    Creates a FeatureUnion that combines multiple parallel transformers into one feature matrix
    This allows for parallel processing of different features.
    """
    return FeatureUnion([
        ("momentum_20", MomemntumTransformer(window=20, price_col="Close")),
        ("momentum_60", MomemntumTransformer(window=60, price_col="Close")),
        ("zscore", ZScoreTransformer(window=20, price_col="Close")),
        # Add more transformers as needed
    ])

def build_feature_pipeline() -> Pipeline:
    """
    For later for chained operations (imputation -> union -> scaling)
    Builds a complete feature extraction pipeline.
    This pipeline can be used to transform raw data into features for modeling.
    """
    return Pipeline([
        ("feature_union", create_feature_union()),
        # ("impute", SimpleImputer()), #e.g. drop or fill NaNs
        # ("scale", StandardScaler()), #e.g. scale features to zero mean and unit variance
        # ("encode", OneHotEncoder()), #e.g. encode categorical variables
        # ("select", SelectKBest(k=10)), #e.g. select top k 
        # Add more steps if needed, e.g., scaling, encoding, etc.
    ])