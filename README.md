# Step 1: Load the real dataset
import pandas as pd
df = pd.read_csv("DistrictwisePensiondataundertheNationalSocialAssistanceProgrammeNSAP.csv")





# Step 2: Define target and input features
y = df['schemecode']
X = df[[
    'districtname', 'totalmale', 'totalfemale', 'totaltransgender',
    'totalsc', 'totalst', 'totalgen', 'totalobc'
]]

# Step 3: Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

categorical_features = ['districtname']
numerical_features = ['totalmale', 'totalfemale', 'totaltransgender', 'totalsc', 'totalst', 'totalgen', 'totalobc']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, numerical_features)
])

# Step 4: Model pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Step 5: Train and evaluate
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
