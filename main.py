from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from functions.functions import *
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv

load_dotenv()

path = os.getenv('DIRECTORY_PATH')

df = pd.read_csv(f"{path}/dataset/anime.csv")

print("\nTABLE DESCRIPTION")
print(df.describe())

print("\COLUMNS")
print(df.columns)

print("\nFIRST LINES OF THE TABLE")
print(df.head())

print("\nTABLE INFORMATION")
print(df.info())


print("\nNUMBER OF NULL VALUES PER COLUMN")
print(df.isnull().sum())



sns.heatmap(df.isnull(), cbar=False)
plt.show()


df_rating_null = df[df['rating'].isnull()]
df_genre_null = df[df['genre'].isnull()]
df_type_null = df[df['type'].isnull()]

print("\nTABLE DATA WITH NULL RATING VALUES")
print(df_rating_null[['name', 'members']].head())
print("\nTABLE DATA WITH NULL GENRE VALUES")
print(df_genre_null[['name', 'members']].head())
print("\nTABLE DATA WITH NULL TYPE VALUES")
print(df_type_null[['name', 'members']].head())






df = drop_na(df)




values = df['episodes'].values
df['episodes'] = pd.DataFrame(values, columns=["episodes"])

df['episodes'] = df['episodes'].map(to_int)

print("\nNUMERICAL VALUES OF EPISODES")
print(df['episodes'])



print("\nNUMBER OF ZEROS IN THE COLUMN episodes")
zero_values = df[df == 0].count(axis=0)
print(zero_values[zero_values > 0])


df = df.replace(0, np.nan)
sns.heatmap(df.isnull(), cbar=False)
plt.show()



df = drop_na(df)


print("\nDESCRIPTION OF THE COLUMNS name, genre and type")
print(df[['name', 'genre', 'type']].describe())



df_best5 = df[['name', 'rating', 'members', 'episodes']]

print("\n5 ANIME WITH THE HIGHEST RATING")
print(df_best5.sort_values(by="rating", ascending=False).head())

print("\n5 ANIME WITH THE LARGEST NUMBER OF MEMBERS")
print(df_best5.sort_values(by="members", ascending=False).head())

print("\n5 ANIMES WITH THE LARGEST NUMBER OF EPISODES")
print(df_best5.sort_values(by="episodes", ascending=False).head())


df = df[['name', 'genre', 'type', 'episodes', 'rating', 'members']]
print("\nDATASET WITHOUT ID COLUMN")
print(df.head())


print("\nDESCRIPTION OF THE COLUMNS episodes, rating and members")
print(df[['episodes', 'rating', 'members']].describe())


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes = axes.flat

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

for i, colum in enumerate(numeric_columns):
    sns.histplot(
        data=df,
        x=colum,
        stat="count",
        kde=True,
        color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
        line_kws={'linewidth': 2},
        alpha=0.3,
        ax=axes[i]
    )
    axes[i].set_title(colum, fontsize=10, fontweight="bold")
    axes[i].tick_params(labelsize=8)
    axes[i].set_xlabel("")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribution of numerical variables', fontsize=10, fontweight="bold")
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

for i, colum in enumerate(numeric_columns):
    axs[i].hist(x=df[colum], bins=20, color="#3182bd", alpha=0.5)
    axs[i].plot(df[colum], np.full_like(df[colum], -0.01), '|k', markeredgewidth=1)
    axs[i].set_title(f'Distribution {colum}')
    axs[i].set_xlabel(colum)
    axs[i].set_ylabel('counts')

plt.tight_layout()
plt.show()





for i, colum in enumerate(numeric_columns):
    print(f"\nCENTRALIZATION MEASURES {colum}")
    print(f'Mean:{df[colum].mean()} \
     \nMedian: {df[colum].median()} \
     \nMode: {df[colum].mode()}')



print(f'\nThe variance is:\n{df.var(numeric_only=True)}')



print(f'\nStandard Deviation per row:\n{df.std(axis=0, numeric_only=True)}')

for i, colum in enumerate(numeric_columns):
    print(f"\nRANGE OF {colum}")
    print(f'The range is: {df[colum].max() - df[colum].min()}')

for i, colum in enumerate(numeric_columns):
    print(f"\nTHE INTERQUARTILE RANGE OF {colum}")
    print(f'The IQR is: {df[colum].quantile(0.75) - df[colum].quantile(0.25)}')

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
print(
    f'\nThe coefficient of variation is:\n{df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"]).apply(cv)}')




print(f"\nAsymmetry measures are:\n{df.skew(numeric_only=True)}")


print(f"\nThe kurtosis measures are:\n{df.kurt(numeric_only=True)}")





all_genres = []
for item in df['genre']:
    item = item.strip()
    all_genres.extend(item.split(', '))

c = Counter(all_genres)

fig, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.countplot(all_genres)

plt.title('Number of series by genre')
plt.xlabel('Genre')
plt.ylabel('Number of series')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


for column in numeric_columns:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    sns.barplot(x='type', y=column, data=df, ax=ax)

    plt.title(f'{column} according to the type of anime')
    plt.xlabel('Type')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()


print("\nLET'S SEE IF THE GROUP IS BALANCED")
print(df.groupby('type').size())



print("\nMEAN AND STANDARD DEVIATION BY GROUP\n")
for column in numeric_columns:
    print(f"\nMEAN AND STANDARD DEVIATION OF {column}")
    print(df.groupby('type')[column].agg(['mean', 'std']).round(2))



fig, ax = plt.subplots(1, 1, figsize=(15, 8))
type_data = Counter(df.type)
labels = list(type_data.keys())
sizes = list(type_data.values())

ax.pie(sizes, labels=labels, shadow=False, startangle=0, autopct="%1.2f%%")
ax.axis('equal')
ax.set_title("Chart by type of anime")
plt.show()




for column in numeric_columns:
    plot_boxplot(df, column)


df_normal = df


df_without_outliers = delete_outliers(df, numeric_columns)

print("\nTABLE WITHOUT OUTLIERS")
print(df_without_outliers.shape)
print(df_without_outliers.info())

for column in numeric_columns:
    plot_boxplot(df_without_outliers, column)



df_replaced_outliers = replace_outliers_mean(df, numeric_columns)

print("\nTABLE WITH REPLACED OUTLIERS")
print(df_replaced_outliers.shape)
print(df_replaced_outliers.info())

for column in numeric_columns:
    plot_boxplot(df_replaced_outliers, column)



df_replaced_outliers = delete_outliers(df_replaced_outliers, numeric_columns)

print("\nTABLE WITH REPLACED OUTLIERS")
print(df_replaced_outliers.shape)
print(df_replaced_outliers.info())

for column in numeric_columns:
    plot_boxplot(df_replaced_outliers, column)



print("\nCORRELATION MATRIX WITH OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df.corr(method=colum, numeric_only=True)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()

print("\nCORRELATION MATRIX WITHOUT OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df_without_outliers.corr(method=colum, numeric_only=True)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()

print("\nCORRELATION MATRIX WITH REPLACED OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df_replaced_outliers.corr(method=colum, numeric_only=True)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()





print("\nNORMAL DATASET PREDICTION")
linear_regression(df)


print("\nDATASET PREDICTION WITHOUT OUTLIERS")
linear_regression(df_without_outliers)


print("\nDATASET PREDICTION WITH REPLACED OUTLIER")
linear_regression(df_replaced_outliers)

print("\nDESCRIPTION OF THE DATASET WITH REPLACED OUTLIER")
print(df_replaced_outliers.describe())




poli_regression(df, 2)
poli_regression(df, 3)
poli_regression(df, 4)
poli_regression(df, 5)


poli_regression(df_without_outliers, 2)
poli_regression(df_without_outliers, 3)
poli_regression(df_without_outliers, 4)

print("\nDATASET DESCRIPTION WITHOUT OUTLIERS")
print(df_without_outliers.describe())


poli_regression(df_replaced_outliers, 2)
poli_regression(df_replaced_outliers, 3)
poli_regression(df_replaced_outliers, 4)
poli_regression(df_replaced_outliers, 5)
poli_regression(df_replaced_outliers, 6)

print("\nDESCRIPTION OF THE DATASET WITH REPLACED OUTLIER")
print(df_replaced_outliers.describe())




regression_tree = DecisionTreeRegressor(random_state=0)
regression_random_forest = RandomForestRegressor(n_estimators=300, random_state=0)

print("\nPREDICTION WITH DECISION TREES IN NORMAL DATASET")
show_fit(regression_tree, 'Decision trees', df)

max_depth = 3
for i in range(7):
    print("\nDEPTH ", max_depth)
    regression_tree_limited = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    show_fit(regression_tree_limited, 'Decision trees', df)
    max_depth = max_depth + 1


print("\nPREDICTION WITH RANDOM FOREST IN NORMAL DATASET")
show_fit(regression_random_forest, 'Random Forest', df)



print("\nPREDICTION WITH DECISION TREES IN DATASET WITHOUT OUTLIERS")
show_fit(regression_tree, 'Decision trees', df_without_outliers)

max_depth = 3
for i in range(2):
    print("\nDEPTH ", max_depth)
    regression_tree_limited = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    show_fit(regression_tree_limited, 'Decision trees', df_without_outliers)
    max_depth = max_depth + 1


print("\nPREDICTION WITH RANDOM FOREST IN DATASET WITHOUT OUTLIERS")
show_fit(regression_random_forest, 'Random Forest', df_without_outliers)


max_depth = 3
for i in range(4):
    print("\nDEPTH ", max_depth)
    random_forest_limited = RandomForestRegressor(n_estimators = 300, random_state = 0, max_depth=max_depth)
    show_fit(random_forest_limited, 'Random Forest', df_without_outliers)
    max_depth = max_depth + 1



print("\nPREDICTION WITH DECISION TREES IN DATASET WITH REPLACED OUTLIERS")
show_fit(regression_tree, 'Decision trees', df_replaced_outliers)

print("\nPREDICTION WITH RANDOM FOREST IN DATASET WITH REPLACED OUTLIERS")
show_fit(regression_random_forest, 'Random Forest', df_replaced_outliers)