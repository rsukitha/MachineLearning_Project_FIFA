import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.ensemble import RakelD
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from sklearn.svm import SVC
from keras import Sequential
from keras.layers import Dense

# Constants
knr_n_neighbours = 3
rf_xgb_n_estimators = 100
no_imp_features_to_use = 10
accepted_error_rate_price_imputation = 0.25

# Load the FIFA-18 dataset.
df = pd.read_csv("FIFA18 - Ultimate Team Players.csv", na_values="N/A")

# EDA
# print(df.describe())
# print(df.isnull().sum())

# Drop FIFA icons, player of the week, etc types of duplicates. Keep only the normal.
df = df[df.revision == "Normal"]

# Remove columns that will have no impact on the target.
useless_cols = ["player_ID", "player_name", "player_extended_name", "revision", "origin", "club", "league",
                "nationality", "date_of_birth", "added_date"]
df = df.drop(useless_cols, axis=1)

# Rename price_ps4 column to price
df = df.rename(columns={'price_ps4': 'price'})

# Convert Quality (categorical variable) to points. Bronze = 1 to Gold - Rare = 6.
df.quality = df.quality.map({
    'Bronze': 1,
    'Bronze - Rare': 2,
    'Silver': 3,
    'Silver - Rare': 4,
    'Gold': 5,
    'Gold - Rare': 6
})

# Fill missing values with 0 assuming GKs will not have abilities that other players have and vice versa.
df = df.fillna(0)

# One Hot Encode Categorical columns.
categorical_cols = ["pref_foot", "att_workrate", "def_workrate"]
categorical_cols_df = pd.get_dummies(df[categorical_cols])

# Split comma separated values in columns traits and specialities into separate values and One Hot Encode all of them.
traits_df = df.traits.str.get_dummies(',')
specialties_df = df.specialties.str.get_dummies(',')

# Drop original categorical columns and join the newly created columns with the rest of the data frame.
categorical_cols.extend(['traits', 'specialties'])
df = df.drop(categorical_cols, axis=1)
df: pd.DataFrame = pd.concat([df, categorical_cols_df, traits_df], axis=1, join="inner")

# Get the records for which price has to be imputed.
X_observed = df[df.price != 0]
X_missing = df[df.price == 0].drop(['price'], axis=1)

X_observed_train, X_observed_test = train_test_split(X_observed, test_size=0.25, random_state=42)

# Create a position map
position_map = {}
positions = X_observed.position.unique()
for i in range(len(positions)):
    position_map[i] = positions[i]

# Declare the position feature importance map
position_feature_importance_map = {}

# Create a list of models to compare and select the best model to use for imputing price with values of 0.
models = [KNeighborsRegressor(n_neighbors=knr_n_neighbours),
          RandomForestRegressor(n_estimators=rf_xgb_n_estimators),
          XGBRegressor(n_estimators=rf_xgb_n_estimators, max_depth=7)]

# Declare the subset models map
sub_models_map = {}

# Get all the imputations predicted by each regressor.
all_imputations = []
reordered_y_train = []
reordered_y_test = []
for i in range(len(models)):
    model_imputations = []
    for position in positions:
        # Create subsets by using position as price varies by the player's playing position.
        sub_X_observed_train = X_observed_train[X_observed_train.position == position]
        sub_X_observed_test = X_observed_test[X_observed_test.position == position]
        if sub_X_observed_train.shape[0] != 0:
            X_train = sub_X_observed_train.drop(['price', 'position'], axis=1)
            y_train = sub_X_observed_train.price
            X_test = sub_X_observed_test.drop(['price', 'position'], axis=1)
            y_test = sub_X_observed_test.price
            # Reorder the y value only once.
            if i == 0:
                reordered_y_train.append(y_train.values)
                reordered_y_test.append(y_test.values)
            sub_model = clone(models[i])
            if isinstance(sub_model, KNeighborsRegressor) and knr_n_neighbours > X_train.shape[0]:
                sub_model.n_neighbors = X_train.shape[0]
            sub_model.fit(X_train, y_train)
            # Get the top 10 important features for this model.
            if hasattr(sub_model, 'feature_importances_'):
                imp_features = sub_model.feature_importances_.argsort()[::-1][:no_imp_features_to_use]
                X_train = X_train.iloc[:, imp_features]
                X_test = X_test.iloc[:, imp_features]
            else:
                imp_features = None
            # Trim the training data set with only the important features
            # Fit the model with the new trimmed dataset.
            X_train = X_train.as_matrix()
            X_test = X_test.as_matrix()
            sub_model.fit(X_train, y_train)
            # Store the sub model in the sub model map for each position.
            if position in sub_models_map:
                sub_models_map[position].append(sub_model)
            else:
                sub_models_map[position] = [sub_model]
            # Store the feature importance in the feature importance map for each position.
            if position in position_feature_importance_map:
                position_feature_importance_map[position].append(imp_features)
            else:
                position_feature_importance_map[position] = [imp_features]

            model_imputations.append(sub_model.predict(X_test))
    all_imputations.append(model_imputations)

# Score how each regressor performs for every subset of the data.
scores = {}
for model_imputation in all_imputations:
    for i in range(len(model_imputation)):
        sub_model_imputation = model_imputation[i]
        sub_y_train = reordered_y_test[i]
        count = 0
        for j in range(len(sub_y_train)):
            actual = sub_y_train[j]
            imputation = sub_model_imputation[j]
            if 1 - (min(imputation, actual) / max(imputation, actual)) < accepted_error_rate_price_imputation:
                count += 1
        score = (count / len(sub_y_train)) * 100
        if i in scores:
            scores[i].append(score)
        else:
            scores[i] = [score]

# Replace position index with position name in scores map
scores = dict((position_map[key], value) for (key, value) in scores.items())

avg_scores = []
for i in range(len(models)):
    score = 0
    for pos_scores in scores.values():
        score += pos_scores[i]
    avg_scores.append(score / len(scores.values()))
print("Avg scores = {}".format(avg_scores))

# # Check what accuracy we get for the entire dataset.
# c = 0
# percent_completed = 0
# index_percent_count = 0
# for i, X in X_observed_test.iterrows():
#     best_model_index = 0
#     best_score = 0
#     index = 0
#     if X.position in scores:
#         for score in scores[X.position]:
#             if score > best_score:
#                 best_score = score
#                 best_model_index = index
#             index += 1
#         best_model = sub_models_map[X.position][best_model_index]
#         X_impute = X
#         best_features = position_feature_importance_map[X.position][best_model_index]
#         if best_features is not None:
#             columns = X_observed_test.drop(['price', 'position'], axis=1).columns[best_features]
#             X_impute = X_impute[columns].as_matrix()
#             X_impute = X_impute.reshape(1, -1)
#         else:
#             X_impute = X_impute.drop(['position'])
#         # X_impute = X_impute.reshape(1, -1)
#         price = best_model.predict(X_impute)[0]
#         actual = X.price
#         if 1 - (min(price, actual) / max(price, actual)) < accepted_error_rate_price_imputation:
#             c += 1
#     index_percent_count += 1
#     if index_percent_count >= X_observed_test.shape[0] / 100:
#         percent_completed += 1
#         index_percent_count = 0
#         print("{}% finished..".format(percent_completed))
# print("Accuracy = {}".format((c / X_observed_test.shape[0]) * 100))

# Select the best model and fill in the missing price value for the dataset
for i, X in X_missing.iterrows():
    best_model_index = 0
    best_score = 0
    index = 0
    if X.position in scores:
        for score in scores[X.position]:
            if score > best_score:
                best_score = score
                best_model_index = index
            index += 1
        best_model = sub_models_map[X.position][best_model_index]
        X_impute = X
        best_features = position_feature_importance_map[X.position][best_model_index]
        if best_features is not None:
            columns = X_missing.drop(['position'], axis=1).columns[best_features]
            X_impute = X_impute[columns].as_matrix()
            X_impute = X_impute.reshape(1, -1)
        else:
            X_impute = X_impute.drop(['position'])
        X_impute = X_impute.reshape(1, -1)
        price = best_model.predict(X_impute)[0]
        df.loc[i, 'price'] = price
    else:
        # Drop rows for which we do not have training model
        df = df.drop([i])

# Check how well a regressor works for the entire dataset.
count = 0
regressor = XGBRegressor(n_estimators=100)
y_observed_train = X_observed_train.price
X_observed_train = X_observed_train.drop(['position', 'price'], axis=1)
regressor.fit(X_observed_train, y_observed_train)
y_observed_test = X_observed_test.price
X_observed_test = X_observed_test.drop(['position', 'price'], axis=1)
predictions = regressor.predict(X_observed_test)
for i in range(y_observed_test.shape[0]):
    actual = y_observed_test.values[i]
    prediction = predictions[i]
    if 1 - (min(actual, prediction) / max(actual, prediction)) < accepted_error_rate_price_imputation:
        count += 1
print("Accuracy = {}".format((count / y_observed_test.shape[0]) * 100))

# Create the training and test data
X = df.drop(np.append(traits_df.columns.values, 'position'), axis=1)
y = df[traits_df.columns.values]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create the classifier
# classifier = LabelPowerset(LogisticRegression())

# classifier = MLkNN(k=3)
# X_train = lil_matrix(X_train).toarray()
# y_train = lil_matrix(y_train).toarray()
# X_test = lil_matrix(X_test).toarray()

# classifier = ClassifierChain(RandomForestClassifier(n_estimators=100, n_jobs=10))

# classifier = RakelD(
#     base_classifier=RandomForestClassifier(n_estimators=100, n_jobs=10),
#     base_classifier_require_dense=[True, False],
#     labelset_size=61
# )

# X_test = X_test.as_matrix()

# classifier = LabelPowerset(RandomForestClassifier(n_estimators=100, n_jobs=10))
#
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)
# print("Accuracy = {}".format(accuracy_score(y_test, predictions)))

imp_features_map = {}
rfc_models = []
for i in range(y_train.shape[1]):
    column = y_train.columns.values[i]
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train[column])
    imp_features = rfc.feature_importances_.argsort()[::-1][:no_imp_features_to_use]
    X_train_trimmed = X_train.iloc[:, imp_features]
    rfc.fit(X_train_trimmed, y_train[column])
    imp_features_map[i] = imp_features
    rfc_models.append(rfc)
    print("{} random forests trained..".format(i + 1))

score = 0
exacts = 0
c = 0
pc = 0
for i, X in X_test.iterrows():
    exact = True
    for j in range(len(rfc_models)):
        rfc = rfc_models[j]
        actual = y_test[y_test.columns.values[j]][i]
        prediction = rfc.predict(X[imp_features_map[j]].values.reshape(1, -1))
        if actual == prediction:
            score += 1
        elif exact is True:
            exact = False
    if exact:
        exacts += 1
    c += 1
    if c >= X_test.shape[0] / 100:
        c = 0
        pc += 1
        print("{}% completed..".format(pc))
print("Accuracy = {}".format((score / (X_test.shape[0] * y_test.shape[1])) * 100))
print("Exact Matches = {}".format((exacts / X_test.shape[0]) * 100))

# neural_nets = []
# for i in range(y_train.shape[1]):
#     nn = Sequential()
#     nn.add(Dense(8, input_dim=X_train.shape[1], activation="relu"))
#     nn.add(Dense(8, activation="relu"))
#     nn.add(Dense(1, activation="sigmoid"))
#     nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     column = y_train.columns.values[i]
#     nn.fit(X_train, y_train[column], epochs=10, batch_size=10, verbose=0)
#     neural_nets.append(nn)
#     print("{} neural networks trained..".format(i + 1))
#
# score = 0
# for i, X in X_test.iterrows():
#     for j in range(len(neural_nets)):
#         nn = neural_nets[j]
#         actual = y_test[y_test.columns.values[j]][i]
#         prediction = nn.predict(X.values.reshape(1, -1))
#         if actual == prediction:
#             score += 1
#
# print("Accuracy = {}".format((score / (X_test.shape[0] * y_test.shape[1])) * 100))
