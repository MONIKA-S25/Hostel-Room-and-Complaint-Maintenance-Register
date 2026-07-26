import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset created in Step 1
df = pd.read_csv('dataset.csv')

# Filter records with active complaints
df = df[df['complaint_type'] != 'None'].copy()

# Convert text variables into numeric codes
df['complaint_code'] = df['complaint_type'].astype('category').cat.codes
df['block_code'] = df['block'].astype('category').cat.codes

# Define inputs (X) and target output (y)
X = df[['complaint_code', 'block_code']]
y = df['is_delayed']

# Split dataset into training and test sets (fixed random seed = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train standard Decision Tree classifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save trained model to file
joblib.dump(model, 'delay_model.pkl')
print("Success! Model trained and saved as delay_model.pkl.")