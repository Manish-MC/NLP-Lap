import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

docs = []

n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input(f"Enter document {i + 1}: "))

query = input("\nEnter search query: ")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)      # <-- Fixed
query_vec = vectorizer.transform([query])

# Cosine Similarity
scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")
for i, s in enumerate(scores[0]):
    print(f"Document {i+1}: {round(s,3)}")

# LSA using Truncated SVD
n_components = min(2, X.shape[0]-1, X.shape[1]-1)

if n_components >= 1:
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    X_lsa = svd.fit_transform(X)
    query_lsa = svd.transform(query_vec)

    lsa_scores = cosine_similarity(query_lsa, X_lsa)

    print("\nLSA Similarity Scores:")
    for i, s in enumerate(lsa_scores[0]):
        print(f"Document {i+1}: {round(s,3)}")

    best = np.argmax(lsa_scores)

    print("\nMost Relevant Document:")
    print(docs[best])

else:
    print("\nNot enough documents/features to perform LSA.")