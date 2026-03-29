import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Crop_recommendation.csv')

# data.head()

# data.isnull().sum()

# Split features and labels-
X = data.iloc[:,:-1] #features
y = data.iloc[:,-1] #labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model-
model = RandomForestClassifier()


model.fit(X_train, y_train)

# Make Predictions-
# predict = model.predict(X_test)

pickle.dump(model, open("model.pkl", "wb"))
# Evaluate Model
#accuracy = model.score(X_test, y_test)
#print(f'Accuracy : {accuracy}')

# input data-
#feature = [[36, 58, 25, 28.66024, 59.31891, 8.399136, 36.9263]]
#predicted_crop = model.predict(feature)
#print(f"Predicted Crop : {predicted_crop}")
