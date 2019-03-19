import pandas as pd


def preprocess_data():
    # Load the FIFA-18 dataset.
    df = pd.read_csv("FIFA18 - Ultimate Team players.csv", na_values="N/A")

    # EDA
    print(df.describe())
    print(df.isnull().sum())

    # Drop FIFA icons, player of the week, etc types of duplicates. Keep only the normal.
    df = df[df.revision == "Normal"]

    # Remove columns that will have no impact on the target.
    useless_cols = ["player_ID", "player_name", "player_extended_name", "revision", "origin", "club", "league",
                    "nationality", "date_of_birth", "added_date", "specialties"]
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

    # Split comma separated values in columns "traits" into separate values and One Hot Encode all of them.
    traits_df = df.traits.str.get_dummies(',')

    # Drop original categorical columns and join the newly created columns with the rest of the data frame.
    # categorical_cols.extend(['traits'])
    df = df.drop(categorical_cols, axis=1)
    return pd.concat([df, categorical_cols_df, traits_df], axis=1, join="inner"), traits_df.columns.values


def get_preprocessed_dataframe():
    df, traits_columns = preprocess_data()
    return df, traits_columns
