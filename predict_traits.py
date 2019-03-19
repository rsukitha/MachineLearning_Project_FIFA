from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from keras import Sequential
from keras.layers import Dense
from pre_process import get_preprocessed_dataframe

# Constants
estimators = 1
epochs = 10
no_imp_features_to_use = 10


def impute_with_labelpowerset():
    x_observed_train, x_observed_test, y_observed_train, y_observed_test, missing, df, traits_columns = get_data()
    labelpowerset(x_observed_train, y_observed_train, x_observed_test, y_observed_test)
    return df


def impute_with_ann():
    x_observed_train, x_observed_test, y_observed_train, y_observed_test, missing, df, traits_columns = get_data()
    ann_chain = ann(x_observed_train, y_observed_train)
    print("Imputation Accuracy = {} ".format(score_ann(ann_chain, x_observed_test, y_observed_test)))
    impute_missing_ann(ann_chain, missing, df, traits_columns)
    df = df.drop(['traits'], axis=1)
    df.to_pickle("traits.df")
    return df


def impute_with_rf():
    x_observed_train, x_observed_test, y_observed_train, y_observed_test, missing, df, traits_columns = get_data()
    imp_features_map = get_imp_features(x_observed_train, y_observed_train)
    rf_chain = rf(x_observed_train, y_observed_train, imp_features_map)
    print("Imputation Accuracy = {} ".format(score_rf(rf_chain, x_observed_test, y_observed_test, imp_features_map)))
    impute_missing_rf(rf_chain, missing, df, imp_features_map, traits_columns)
    df = df.drop(['traits'], axis=1)
    df.to_pickle("traits.df")
    return df


def get_imp_features(x_train, y_train):
    imp_features_map = {}
    for i in range(y_train.shape[1]):
        column = y_train.columns.values[i]
        rfc = RandomForestClassifier(n_estimators=estimators)
        rfc.fit(x_train, y_train[column])
        imp_features = rfc.feature_importances_.argsort()[::-1][:no_imp_features_to_use]
        imp_features_map[i] = imp_features
    return imp_features_map


def get_data():
    df, traits_columns = get_preprocessed_dataframe()
    observed = df[df.traits != 0].drop(['position', 'traits'], axis=1)
    missing = df[df.traits == 0].drop(['position', 'traits'], axis=1)
    x_observed = observed.drop(traits_columns, axis=1)
    y_observed = observed[traits_columns]
    x_observed_train, x_observed_test, y_observed_train, y_observed_test = train_test_split(x_observed, y_observed,
                                                                                            test_size=0.25)
    return x_observed_train, x_observed_test, y_observed_train, y_observed_test, missing, df, traits_columns


def labelpowerset(x_train, y_train, x_test, y_test):
    classifier = LabelPowerset(RandomForestClassifier(n_estimators=estimators))
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    print("Accuracy = {}".format(accuracy_score(y_test, predictions)))


def rf(x_train, y_train, imp_features_map):
    rfc_models = []
    for i in range(y_train.shape[1]):
        column = y_train.columns.values[i]
        rfc = RandomForestClassifier(n_estimators=estimators)
        x_train_trimmed = x_train.iloc[:, imp_features_map[i]]
        rfc.fit(x_train_trimmed, y_train[column])
        rfc_models.append(rfc)
        print("{} random forests trained..".format(i + 1))
    return rfc_models


def ann(x_train, y_train):
    neural_nets = []
    for i in range(y_train.shape[1]):
        nn = Sequential()
        nn.add(Dense(8, input_dim=x_train.shape[1], activation="relu"))
        nn.add(Dense(8, activation="relu"))
        nn.add(Dense(1, activation="sigmoid"))
        nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        column = y_train.columns.values[i]
        nn.fit(x_train, y_train[column], epochs=epochs, batch_size=10, verbose=0)
        neural_nets.append(nn)
        print("{} neural networks trained..".format(i + 1))
    return neural_nets


def score_rf(rfc_models, x_test, y_test, imp_features_map):
    score = 0
    exacts = 0
    c = 0
    pc = 0
    for i, X in x_test.iterrows():
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
        if c >= x_test.shape[0] / 100:
            c = 0
            pc += 1
            print("{}% completed..".format(pc))
    return (score / (x_test.shape[0] * y_test.shape[1])) * 100, (exacts / x_test.shape[0]) * 100


def score_ann(neural_nets, x_test, y_test):
    score = 0
    exacts = 0
    c = 0
    pc = 0
    for i, X in x_test.iterrows():
        exact = True
        for j in range(len(neural_nets)):
            nn = neural_nets[j]
            actual = y_test[y_test.columns.values[j]][i]
            prediction = nn.predict(X.values.reshape(1, -1))
            if actual == prediction:
                score += 1
            elif exact is True:
                exact = False
        if exact:
            exacts += 1
        c += 1
        if c >= x_test.shape[0] / 100:
            c = 0
            pc += 1
            print("{}% completed..".format(pc))
    return (score / (x_test.shape[0] * y_test.shape[1])) * 100, (exacts / x_test.shape[0]) * 100


def impute_missing_rf(rfc_models, missing, df, imp_features_map, traits_columns):
    c = 0
    pc = 0
    for i, X in missing.iterrows():
        for j in range(len(rfc_models)):
            model = rfc_models[j]
            prediction = model.predict(X[imp_features_map[j]].values.reshape(1, -1))
            df.loc[i, traits_columns[j]] = prediction
        c += 1
        if c >= missing.shape[0] / 100:
            c = 0
            pc += 1
            print("{}% completed..".format(pc))


def impute_missing_ann(ann_models, missing, df, traits_columns):
    c = 0
    pc = 0
    for i, X in missing.iterrows():
        for j in range(len(ann_models)):
            model = ann_models[j]
            prediction = model.predict(X.values.reshape(1, -1))
            df.loc[i, traits_columns[j]] = prediction
        c += 1
        if c >= missing.shape[0] / 100:
            c = 0
            pc += 1
            print("{}% completed..".format(pc))
