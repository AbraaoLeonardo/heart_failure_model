from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from prediction_model.config import config

classification_pipeline = Pipeline([
    ("StandardScaler", StandardScaler()),
    ("RandomForestClassifier",RandomForestClassifier(random_state=config.RANDOM_SEED,
                                                     n_estimators=config.N_ESTIMATORS,
                                                     max_depth=config.MAX_DEPTH,
                                                     criterion=config.CRITERION,
                                                     max_leaf_nodes=config.MAX_LEAF_NODES))]
)
