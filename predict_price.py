import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from predict_traits import impute_with_rf
import os.path

# Constants
knr_n_neighbours = 3
estimators = 100
no_imp_features_to_use = 10
accepted_error_rate_price_imputation = 0.40
# to_drop_cols = ['position', 'price', 'price_xbox', 'price_pc']
to_drop_cols = ['position', 'price']

# Maps
position_map = {}
position_feature_importance_map = {}
sub_models_map = {}


def get_data():
    if os.path.isfile("traits.df"):
        df = pd.read_pickle("traits.df")
    else:
        df = impute_with_rf()
    # Fill missing price values from xbox and pc price values
    df.price = df.price.fillna(df.price_xbox).fillna(df.price_pc)
    df_observed = df[df.price != 0]
    x = df_observed.drop(to_drop_cols, axis=1)
    y = df_observed.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return x_train, x_test, y_train, y_test, df_observed


def create_position_map(x):
    positions = x.position.unique()
    for i in range(len(positions)):
        position_map[i] = positions[i]


def create_models():
    models = [KNeighborsRegressor(n_neighbors=knr_n_neighbours),
              RandomForestRegressor(n_estimators=estimators),
              XGBRegressor(n_estimators=estimators, max_depth=20)]
    return models


def train(df_train, models):
    # Get all the imputations predicted by each regressor.
    for i in range(len(models)):
        for position in df_train.position.unique():
            # Create subsets by using position as price varies by the player's playing position.
            sub_df_train = df_train[df_train.position == position]
            x_train = sub_df_train.drop(to_drop_cols, axis=1)
            y_train = sub_df_train.price
            sub_model = clone(models[i])
            if isinstance(sub_model, KNeighborsRegressor) and knr_n_neighbours > x_train.shape[0]:
                sub_model.n_neighbors = x_train.shape[0]
            sub_model.fit(x_train, y_train)
            # Get the top 10 important features for this model.
            if hasattr(sub_model, 'feature_importances_'):
                imp_features = sub_model.feature_importances_.argsort()[::-1][:no_imp_features_to_use]
                x_train = x_train.iloc[:, imp_features]
                imp_feature_names = x_train.columns.values
            else:
                imp_feature_names = None
            # Trim the training data set with only the important features
            # Fit the model with the new trimmed dataset.
            x_train = x_train.as_matrix()
            sub_model.fit(x_train, y_train)
            # Store the sub model in the sub model map for each position.
            if position in sub_models_map:
                sub_models_map[position].append(sub_model)
            else:
                sub_models_map[position] = [sub_model]
            # Store the feature importance in the feature importance map for each position.
            if position in position_feature_importance_map:
                position_feature_importance_map[position].append(imp_feature_names)
            else:
                position_feature_importance_map[position] = [imp_feature_names]


def score_models_for_position(df_test):
    scores_map = {}
    for position in df_test.position.unique():
        sub_df_test = df_test[df_test.position == position]
        x_test = sub_df_test.drop(to_drop_cols, axis=1)
        y_test = sub_df_test.price
        if position in sub_models_map:
            sub_models = sub_models_map[position]
            pos_imp_features = position_feature_importance_map[position]
            for i in range(len(sub_models)):
                sub_model = sub_models[i]
                imp_features = pos_imp_features[i]
                if imp_features is not None:
                    x_test_trimmed = x_test[imp_features]
                    x_test_trimmed = x_test_trimmed.as_matrix()
                    predictions = sub_model.predict(x_test_trimmed)
                else:
                    predictions = sub_model.predict(x_test)
                score = score_predictions(predictions, y_test.values)
                if position in scores_map:
                    scores_map[position].append(score)
                else:
                    scores_map[position] = [score]
    return scores_map


def get_best_models_for_position(scores_map):
    best_models = {}
    for position, model_scores in scores_map.items():
        max_score = 0
        for i in range(len(model_scores)):
            if model_scores[i] > max_score:
                max_score = model_scores[i]
                best_models[position] = sub_models_map[position][i], position_feature_importance_map[position][i]
    return best_models


def predict_xgb(x_train, x_test, y_train, y_test):
    regressor = XGBRegressor(n_estimators=estimators, max_depth=20)
    regressor.fit(x_train, y_train)
    columns = x_train.columns[regressor.feature_importances_.argsort()[::-1][:no_imp_features_to_use]]
    x_train_trimmed = x_train[columns]
    x_test_trimmed = x_test[columns]
    regressor.fit(x_train_trimmed, y_train)
    predictions = regressor.predict(x_test_trimmed)
    score = score_predictions(predictions, y_test.values)
    return score


def predict_rf(x_train, x_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators=estimators)
    regressor.fit(x_train, y_train)
    columns = x_train.columns[regressor.feature_importances_.argsort()[::-1][:no_imp_features_to_use]]
    x_train_trimmed = x_train[columns]
    x_test_trimmed = x_test[columns]
    regressor.fit(x_train_trimmed, y_train)
    predictions = regressor.predict(x_test_trimmed)
    score = score_predictions(predictions, y_test.values)
    return score


def score_predictions(predictions, actual_values):
    count = 0
    for i in range(len(actual_values)):
        actual = actual_values[i]
        prediction = predictions[i]
        if 1 - (min(actual, prediction) / max(actual, prediction)) < accepted_error_rate_price_imputation:
            count += 1
    score = (count / len(actual_values)) * 100
    return score


def combined_prediction_position_wise(df, df_validation):
    models = create_models()
    df_train, df_test = train_test_split(df, test_size=0.10)
    train(df_train, models)
    scores_map = score_models_for_position(df_test)
    best_models = get_best_models_for_position(scores_map)
    predictions = []
    actual_values = []
    for i, X in df_validation.iterrows():
        if X.position in best_models:
            best_model, imp_features = best_models[X.position]
            x = X.drop(to_drop_cols)
            if imp_features is not None:
                x_test_trimmed = x[imp_features].as_matrix()
                x_test_trimmed = x_test_trimmed.reshape(1, -1)
                predictions.append(best_model.predict(x_test_trimmed)[0])
            else:
                x = x.values.reshape(1, -1)
                predictions.append(best_model.predict(x)[0])
            actual_values.append(X.price)
    score = score_predictions(predictions, actual_values)
    return score


def predict_all():
    x_train, x_test, y_train, y_test, df = get_data()
    df, df_validation = train_test_split(df, test_size=0.15)
    print("Accuracy using Combination = {}".format(combined_prediction_position_wise(df, df_validation)))
    print("Accuracy using RandomForests = {}".format(predict_rf(x_train, x_test, y_train, y_test)))
    print("Accuracy using XGBoost = {}".format(predict_xgb(x_train, x_test, y_train, y_test)))
