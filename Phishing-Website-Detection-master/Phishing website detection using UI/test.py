import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the datasets
legitimate_urls = pd.read_csv("legitimate-urls.csv")
phishing_urls = pd.read_csv("phishing-urls.csv")
urls = pd.concat([legitimate_urls, phishing_urls])

# Drop unnecessary columns
urls = urls.drop(urls.columns[[0, 3, 5]], axis=1)

# Shuffle the rows
urls = urls.sample(frac=1).reset_index(drop=True)

# Encode categorical variables
encoder = LabelEncoder()
urls['Protocol'] = encoder.fit_transform(urls['Protocol'])

# Feature Engineering - Adding Domain Length as a feature
def extract_domain_length(url):
    parts = url.split('/')
    if len(parts) > 2:
        return len(parts[2].split('.')[0])
    else:
        return 0

urls['Domain_Length'] = urls['Domain'].apply(extract_domain_length)
urls = urls.drop('Domain', axis=1)

# Splitting the data into features and labels
urls_without_labels = urls.drop('label', axis=1)
labels = urls['label']

# Splitting the data into train and test sets
data_train, data_test, labels_train, labels_test = train_test_split(urls_without_labels, labels, test_size=0.20, random_state=100)

# Feature Scaling
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_test_scaled = scaler.transform(data_test)

# Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(data_train, labels_train)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=100)
svm_model.fit(data_train_scaled, labels_train)

# Now let's create some sample new data for demonstration
new_data = pd.DataFrame({
    'Having_@_symbol': [0, 1],
    'Path': [0, 1],
    'Prefix_suffix_separation': [0, 1],
    'Protocol': [0, 1],
    'URL_Length': [0, 1],
    'age_domain': [0, 1],
    'dns_record': [0, 1],
    'domain_registration_length': [0, 1],
    'http_tokens': [0, 1],
    'statistical_report': [0, 1],
    'tiny_url': [0, 1],
    'web_traffic': [0, 1],
    'Having_IP': [0, 1],  # Include the missing features
    'Redirection_//_symbol': [0, 1],  # Include the missing features
    'Sub_domains': [0, 1]  # Include the missing features
})

# Ensure the new data has the same columns as the training data
new_data = new_data[urls_without_labels.columns]

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Make predictions using the models
nb_new_pred = nb_model.predict(new_data)
svm_new_pred = svm_model.predict(new_data_scaled)

# Combine predictions using the ensemble
ensemble_new_pred = []
for i in range(len(new_data)):
    vote = nb_new_pred[i] + svm_new_pred[i]
    if vote >= 1:
        ensemble_new_pred.append(1)  # Predict Phishing
    else:
        ensemble_new_pred.append(0)  # Predict Legitimate

# Now ensemble_new_pred contains predictions for the new data
print(ensemble_new_pred)
