{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8746a6c3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy : 0.9931818181818182\n",
      "Predicted Crop : ['mothbeans']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\pcs\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\utils\\validation.py:2691: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "data = pd.read_csv('Crop_recommendation.csv')\n",
    "\n",
    "data.head()\n",
    "\n",
    "data.isnull().sum()\n",
    "\n",
    "# Split features and labels-\n",
    "X = data.iloc[:,:-1] #features\n",
    "y = data.iloc[:,-1] #labels\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train Model-\n",
    "model = RandomForestClassifier()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Make Predictions-\n",
    "predict = model.predict(X_test)\n",
    "\n",
    "accuracy = model.score(X_test, y_test)\n",
    "print(f'Accuracy : {accuracy}')\n",
    "\n",
    "# input data-\n",
    "feature = [[36, 58, 25, 28.66024, 59.31891, 8.399136, 36.9263]]\n",
    "predicted_crop = model.predict(feature)\n",
    "print(f\"Predicted Crop : {predicted_crop}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
