
"""Attempt to use ML to predict the next location of a device based on its previous locations."""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the data
filepath = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/merged_player_positions.csv"
data = pd.read_csv(filepath)

# Drop the timestamp column
data = data.drop(columns=["timestamp"])

# create new df w/ trilateration data

