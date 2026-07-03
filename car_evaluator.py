import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# GENERATE MOCK DATA (UCI Style)
np.random.seed(42)
num_cars = 1500

# The standard dataset contains purely categorical, ordinal attributes
data = {
    'buying_price': np.random.choice(['low', 'med', 'high', 'vhigh'], num_cars),
    'maint_cost': np.random.choice(['low', 'med', 'high', 'vhigh'], num_cars),
    'doors': np.random.choice(['2', '3', '4', '5more'], num_cars),
    'persons': np.random.choice(['2', '4', 'more'], num_cars),
    'lug_boot': np.random.choice(['small', 'med', 'big'], num_cars),
    'safety': np.random.choice(['low', 'med', 'high'], num_cars)
}
df = pd.DataFrame(data)

def evaluate_car(row):
    if row['safety'] == 'low' or row['persons'] == '2':
        return 'unacc'
    elif row['safety'] == 'high' and row['buying_price'] in ['low', 'med'] and row['maint_cost'] in ['low', 'med']:
        return 'vgood'
    elif row['safety'] in ['med', 'high'] and row['buying_price'] != 'vhigh':
        return 'acc'
    else:
        return 'good'

df['car_acceptability'] = df.apply(evaluate_car, axis=1)
print("--- Car Dataset Sample ---")
print(df.head(), "\n")

# SPLIT FEATURES & TARGET
X = df.drop(columns=['car_acceptability'])
y = df['car_acceptability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ORDINAL CATEGORICAL ENCODING
# Defining the strict order of categorical hierarchies explicitly
price_order = ['low', 'med', 'high', 'vhigh']
maint_order = ['low', 'med', 'high', 'vhigh']
doors_order = ['2', '3', '4', '5more']
persons_order = ['2', '4', 'more']
lug_order = ['small', 'med', 'big']
safety_order = ['low', 'med', 'high']

encoder = OrdinalEncoder(categories=[price_order, maint_order, doors_order, persons_order, lug_order, safety_order])

# Transform features
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Label encode the multi-class target strings into integers (0 to 3)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# TRAIN MULTI-CLASS CLASSIFIER
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_encoded, y_train_encoded)

# EVALUATE PERFORMANCE
y_pred = clf.predict(X_test_encoded)

print("--- Evaluation Results ---")
print(f"Overall Accuracy: {accuracy_score(y_test_encoded, y_pred) * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))