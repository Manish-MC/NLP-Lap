import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download required resource
nltk.download('punkt')

# Relation keywords
keywords = ["treats", "reduces", "controls", "helps"]

# User input
sentence = input("Enter biomedical sentence: ")
actual = int(input("Actual Relation (1/0): "))

# Validate input
if actual not in [0, 1]:
    print("Invalid input! Actual Relation must be 0 or 1.")
    exit()

# Tokenization
tokens = word_tokenize(sentence.lower())

print("\nTokens:")
print(tokens)

# Relation Prediction
predicted = 1 if any(word in keywords for word in tokens) else 0

print("\nPredicted Relation:", predicted)

# Evaluation
y_true = [actual]
y_pred = [predicted]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nEvaluation Metrics")
print("------------------")
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)