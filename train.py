
import pandas as pd
import scipy.sparse
from scipy.sparse import hstack
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from azureml.core import Run

run = Run.get_context()

print("Loading data.....")
df = pd.read_csv("train_data.csv")


# Split the dataset into training and testing sets
X_text = df[['vid_title_preprocessed', 'description_preprocessed']]
X_numeric = df[['publish_day_of_week','publish_hour_of_day','likes','dislikes','comment_count','category_id']]  # Add other numeric features as needed
y = df['views']  # Replace 'views' with the desired target variable

X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(X_text, X_numeric, y, test_size=0.2, random_state=42)

# Apply CountVectorizer on text features
count_vectorizer = CountVectorizer()
title_bow_train = count_vectorizer.fit_transform(X_text_train['vid_title_preprocessed'])
description_bow_train = count_vectorizer.transform(X_text_train['description_preprocessed'])

title_bow_test = count_vectorizer.transform(X_text_test['vid_title_preprocessed'])
description_bow_test = count_vectorizer.transform(X_text_test['description_preprocessed'])

# Apply TfidfVectorizer on text features
tfidf_vectorizer = TfidfVectorizer()
title_tfidf_train = tfidf_vectorizer.fit_transform(X_text_train['vid_title_preprocessed'])
description_tfidf_train = tfidf_vectorizer.transform(X_text_train['description_preprocessed'])

title_tfidf_test = tfidf_vectorizer.transform(X_text_test['vid_title_preprocessed'])
description_tfidf_test = tfidf_vectorizer.transform(X_text_test['description_preprocessed'])

# Concatenate the numeric features with the text features
X_train = scipy.sparse.hstack((title_bow_train, description_bow_train, title_tfidf_train, description_tfidf_train, X_numeric_train))
X_test = scipy.sparse.hstack((title_bow_test, description_bow_test, title_tfidf_test, description_tfidf_test, X_numeric_test))

# Select a predictive model (Random Forest Regressor as an example)
model = RandomForestRegressor(verbose=1)

print("Training Model.....")
# Train the predictive model
model.fit(X_train, y_train)

print("Evaluating Model....")
# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

run.log("Mean Squared Error:", mse)
run.log("Mean Absolute Error:", mae)
run.log("R-squared:", r2)

# saving the model
with open('outputs/model.pkl', 'wb') as file:
    pickle.dump(model, file)


