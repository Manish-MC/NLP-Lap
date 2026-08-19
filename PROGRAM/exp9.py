from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------
# Input Documents and Categories
# -----------------------------------

docs = []
labels = []

n = int(input("Enter number of documents: "))

for i in range(n):
    print(f"\nDocument {i + 1}")
    docs.append(input("Enter document: ").strip())
    labels.append(input("Enter category: ").strip().lower())

# -----------------------------------
# Rule-Based Classification
# -----------------------------------

rule_pred = []

for doc in docs:
    doc_lower = doc.lower()

    if "contract" in doc_lower:
        rule_pred.append("contract")

    elif "judgment" in doc_lower:
        rule_pred.append("judgment")

    else:
        rule_pred.append("agreement")

# Calculate Rule-Based Accuracy
rule_acc = accuracy_score(labels, rule_pred)

# -----------------------------------
# Maximum Entropy Classification
# -----------------------------------

# Convert documents into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# Create Logistic Regression model
# Logistic Regression is commonly used as a Maximum Entropy classifier
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X, labels)

# Predict categories
ml_pred = model.predict(X)

# Calculate accuracy
ml_acc = accuracy_score(labels, ml_pred)

# -----------------------------------
# Display Results
# -----------------------------------

print("\n========== RESULTS ==========")

print("\nActual Categories:")
print(labels)

print("\nRule-Based Predictions:")
print(rule_pred)

print("\nMaximum Entropy Predictions:")
print(list(ml_pred))

print("\nRule-Based Accuracy:", round(rule_acc * 100, 2), "%")
print("Maximum Entropy Accuracy:", round(ml_acc * 100, 2), "%")