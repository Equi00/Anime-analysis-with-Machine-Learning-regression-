import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

def drop_na(data):
    data = data.dropna(axis=0)

    print("\nNUMBER OF NULL VALUES PER COLUMN")
    print(data.isnull().sum())

    sns.heatmap(data.isnull(), cbar=False)
    plt.show()

    return data


def to_int(val):
    try:
        value = int(float(val))
    except ValueError:
        value = 0
    return value


def plot_boxplot(df, ft):
    sns.boxplot(y=ft, data=df)
    plt.show()


def outliers(df, ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr

    ls = df.index[(df[ft] < low) | (df[ft] > up)]

    return ls


def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


def delete_outliers(df, list):
    n = 100
    df_clean = df
    for i in range(n):
        index_list = []
        for feature in list:
            index_list.extend(outliers(df_clean, feature))
        if not index_list:
            break
        df_clean = remove(df_clean, index_list)
    return df_clean


def replace_outliers_mean(df, columns):
    df_clean = df.copy()
    
    for column in columns:
        mean_value = df_clean[column].mean()

        q1 = df_clean[column].quantile(0.25)
        q3 = df_clean[column].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        df_clean[column] = df_clean[column].apply(lambda x: mean_value if x > upper_limit or x < lower_limit else x)
    
    return df_clean


def linear_regression(df):
    df_genre_encoded = pd.get_dummies(df['genre'], prefix='genre')
    df_type_encoded = pd.get_dummies(df['type'], prefix='type')

    # We enter the coded columns
    df_encoded = pd.concat([df, df_genre_encoded], axis=1)
    df_encoded = pd.concat([df_encoded, df_type_encoded], axis=1)

    X = df_encoded.drop(['name', 'genre', 'rating', 'type'], axis=1)  # Predictor variables
    y = df_encoded['rating']  # Target variable

    # Generate our training and test sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("\nWE CHECK TRAINING AND TESTING SET DIMENSIONS")
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    # Fit the RLM model with the training set
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predicting the results on the testing set
    y_pred = model.predict(X_test)

    df_prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df_prediction.head(25)
    print(df1.head())

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def poli_regression(df, number):
    label_encoder = LabelEncoder()

    df_encoded = df

    df_encoded['gennre_encoded'] = label_encoder.fit_transform(df['genre'])
    df_encoded['type_encoded'] = label_encoder.fit_transform(df['type'])

    X = df_encoded[['gennre_encoded', 'type_encoded', 'episodes', 'members']]  # Predictor variables
    y = df_encoded['rating']  # Target variable

    poly_reg = PolynomialFeatures(degree=number)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    y_pred = lin_reg_2.predict(X_poly)

    df['prediction_rating'] = y_pred
    rating_average = df.groupby('type')['rating'].mean()
    prediction_average = df.groupby('type')['prediction_rating'].mean()

    types = rating_average.index
    x = np.arange(len(types))
    bar_width = 0.35

    fig, ax = plt.subplots()
    rating_bars = ax.bar(x - bar_width / 2, rating_average, bar_width, label='Rating average')
    prediction_bars = ax.bar(x + bar_width / 2, prediction_average, bar_width, label='Prediction average')

    ax.set_xlabel('Type')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Average Rating and Average Prediction by Type')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    plt.show()

    print(f"\nREGRESSION GRADE {number}")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))


def show_fit(regressor, title, df):
    label_encoder = LabelEncoder()

    df_encoded = df

    df_encoded['genre_encoded'] = label_encoder.fit_transform(df['genre'])
    df_encoded['type_encoded'] = label_encoder.fit_transform(df['type'])
    X = df_encoded[['genre_encoded', 'type_encoded', 'episodes', 'members']]
    y = df_encoded['rating']

    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    df_prediction = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    df1 = df_prediction.head(25)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.xlabel('Anime Index')
    plt.ylabel('Rating')
    plt.title(f'Rating prediction using {title}')
    plt.legend()
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))

    # To know if there is overfitting we do the following
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor.fit(X_train, y_train)

    cv_scores = cross_val_score(regressor, X_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", mean_cv_score)
    test_score = regressor.score(X_test, y_test)
    print("Score on test data:", test_score)

    # This way we can find out if the cross-validation score values ​​and the test data score differ greatly.
    print("DIFFERENCE: ", test_score - mean_cv_score)  # The difference indicates whether or not there is overfitting of the data.