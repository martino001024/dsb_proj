import os
import kagglehub
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
path = kagglehub.dataset_download("vatsalmavani/spotify-dataset")
print("Path to dataset files:", path)
files = os.listdir(path)
print("Files in dataset:", files)
# Adjust path to point to the 'data' folder
data_path = os.path.join(path, 'data')
# List files in the 'data' folder
files = os.listdir(data_path)
print("Files in 'data' folder:", files)
data = pd.read_csv(os.path.join(data_path, 'data.csv'))
data_by_genres = pd.read_csv(os.path.join(data_path, 'data_by_genres.csv'))
data_by_artist = pd.read_csv(os.path.join(data_path, 'data_by_artist.csv'))
data_by_year = pd.read_csv(os.path.join(data_path, 'data_by_year.csv'))
data_w_genres = pd.read_csv(os.path.join(data_path, 'data_w_genres.csv'))
data.head()
data.shape
data.columns
data.info()
data.describe()
data['explicit'].value_counts()
data['mode'].value_counts()

'''
def get_decade(year):
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

data['decade'] = data['year'].apply(get_decade)

# Set figure size
sns.set(rc={'figure.figsize':(12, 6)})

# Create a colorful vertical countplot
sns.countplot(x='decade', data=data, palette="coolwarm")  # 'viridis' is one color map, but you can choose others like 'cool', 'plasma', etc.

# Adjust labels for better readability
plt.xlabel("Decade")
plt.ylabel("Count")
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.title("Counts by Decade")
plt.show()


feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness',
                 'tempo', 'valence', 'duration_ms', 'explicit', 'key', 'mode', 'year']

X, y = data[feature_names], data['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(8,8)
visualizer.fit(X, y)     # Fit the data to the visualizer
visualizer.show()
data.columns


plt.figure(figsize=(12,12))

correlation_matrix = data[['valence', 'year', 'acousticness', 'danceability',
                           'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key',
                           'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.tight_layout()
plt.show()

'''
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
#sns.boxplot(x='explicit', y='popularity', data=data, palette='coolwarm');
data[data['explicit']==1]['popularity'].quantile(0.75) + 1.5*(data[data['explicit']==1]['popularity'].quantile(0.75) - data[data['explicit']==1]['popularity'].quantile(0.25))
data[data['explicit']==1]['popularity'].nlargest(5)
def outliers(df, ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 = 1.5 * IQR
    ls = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]
    return ls

index_list = []

for feature in ['popularity']:
    index_list.extend(outliers(data, feature))

def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df

cleaned_df = remove(data, index_list)
#sns.boxplot(x='explicit', y='popularity', data = cleaned_df, palette='coolwarm');
#sns.boxplot(x='mode', y='popularity', data=cleaned_df, palette='coolwarm');
len(index_list)
cleaned_df['explicit'].value_counts()
cleaned_df['mode'].value_counts()
data_by_genres.head()
data_by_genres.shape
data_by_genres.columns
top10_genres = data_by_genres.nlargest(10, 'popularity')
#fig = px.bar(top10_genres, x='genres', y=sound_features, barmode='group')
#fig.show()
data_by_artist.head()
data_by_artist.columns
data_by_year.head()
data_by_year.shape
data_by_year.columns
data_by_year.head()


'''
# Create a 3x2 subplot figure
fig = make_subplots(rows=3, cols=2, subplot_titles=sound_features)

# Add each feature to a subplot
for i, feature in enumerate(sound_features):
    row = i // 2 + 1
    col = i % 2 + 1
    fig.add_trace(
        go.Scatter(x=data_by_year['year'], y=data_by_year[feature], mode='lines', name=feature),
        row=row, col=col
    )

# Update layout for better visualization
fig.update_layout(height=600, width=900, title_text="Sound Features Over Time")
fig.show()
'''
data_w_genres.head()
data_w_genres.shape
data_w_genres.columns
#data cleanse
data_by_genres['genres'] = data_by_genres['genres'].apply(lambda x: 'Other' if x == '[]' else x)
data_w_genres['genres'] = data_w_genres['genres'].apply(lambda x: 'Other' if x == '[]' else x)

data = data.drop_duplicates()
data_by_genres = data_by_genres.drop_duplicates()
data_by_artist = data_by_artist.drop_duplicates()
data_by_year = data_by_year.drop_duplicates()
data_w_genres = data_w_genres.drop_duplicates()

data.to_csv(os.path.join(data_path, 'processed_data.csv'), index=False)
data_by_genres.to_csv(os.path.join(data_path, 'processed_data_by_genres.csv'), index=False)
data_by_artist.to_csv(os.path.join(data_path, 'processed_data_by_artist.csv'), index=False)
data_by_year.to_csv(os.path.join(data_path, 'processed_data_by_year.csv'), index=False)
data_w_genres.to_csv(os.path.join(data_path, 'processed_data_w_genres.csv'), index=False)
