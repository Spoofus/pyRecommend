import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import joblib

# Load data from CSV files
click_data = pd.read_csv('click_data.csv') # User behavior data
sales_data = pd.read_csv('sales_data.csv') # Sales history
demographic_data = pd.read_csv('demographic_data.csv') # Age, income, gender
segment_data = pd.read_csv('segment_data.csv') # List of customers in the segment

# Merge data on customer_id
data = pd.merge(click_data, sales_data, on='customer_id')
data = pd.merge(data, demographic_data, on='customer_id')

# Feature engineering
data['click_rate'] = data['clicks'] / data['visits']
data['purchase_frequency'] = data['purchases'] / data['visits']
data['purchased'] = (data['purchases'] > 0).astype(int) # Dummy target variable

# Prepare data for model
features = ['click_rate', 'purchase_frequency', 'age', 'income', 'gender']
X = data[features]
y = data['purchased'] # Target variable indicating if the product was purchased

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')

# Predict for a specific product and segment of customers
product_id = '67890' # Replace with actual product ID
segment_customers = segment_data['customer_id'].unique()
segment_data = data[data['customer_id'].isin(segment_customers)]

# Prepare segment data for prediction
segment_features = segment_data[features]
segment_features_scaled = scaler.transform(segment_features)

# Get top N customers in the segment
segment_data['purchase_probability'] = model.predict_proba(segment_features_scaled)[:, 1]
top_customers = segment_data.sort_values(by='purchase_probability', ascending=False).head(10)['customer_id']
print("Top customers likely to purchase the product:", top_customers)

# Save the top customers to a file
top_customers.to_csv('top_customers.csv', index=False)

# Visualization functions
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure()
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance')
    plt.show()

# Predictions and visualizations
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred_proba)
plot_precision_recall_curve(y_test, y_pred_proba)
plot_feature_importance(model, features)

# Iterate over products and find top 3 products for purchase probability
product_ids = ['12345', '67890', '54321'] # Replace with actual product IDs

top_products = []

for product_id in product_ids:
    segment_customers = segment_data['customer_id'].unique()
    segment_data_product = data[data['customer_id'].isin(segment_customers)]
    
    # Prepare segment data for prediction for each product
    segment_features_product = segment_data_product[features]
    segment_features_scaled_product = scaler.transform(segment_features_product)
    
    # Get purchase probability for each product and find top customers
    segment_data_product['purchase_probability'] = model.predict_proba(segment_features_scaled_product)[:, 1]
    
    top_customers_product = segment_data_product.sort_values(by='purchase_probability', ascending=False).head(3)['customer_id']
    
    top_products.append({
        'product_id': product_id,
        'top_customers': top_customers_product.values.tolist()
    })

# Save the top products and customers to an Excel file
results_df = pd.DataFrame(top_products)
results_df.to_excel('results.xlsx', index=False)

print("Top products and their most likely customers have been saved to results.xlsx.")
