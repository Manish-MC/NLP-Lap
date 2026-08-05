import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

reviews = []

n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input(f"Enter review {i+1}: "))

# Convert text to Bag of Words
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# -----------------------------
# LDA Topic Modeling
# -----------------------------
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\n========== TOPICS ==========")

for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i+1}:")
    top_words = topic.argsort()[-5:][::-1]
    for j in top_words:
        print(words[j])

# -----------------------------
# t-SNE Visualization
# -----------------------------
X_dense = X.toarray()

# Set valid perplexity
perplexity = min(5, len(reviews) - 1)

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42
)

X_tsne = tsne.fit_transform(X_dense)

print("\n========== t-SNE Coordinates ==========")

for i, point in enumerate(X_tsne):
    print(f"Review {i+1}: ({point[0]:.2f}, {point[1]:.2f})")

# Plot
plt.figure(figsize=(7,5))
plt.scatter(X_tsne[:,0], X_tsne[:,1], s=80)

for i in range(len(reviews)):
    plt.text(X_tsne[i,0], X_tsne[i,1], f"R{i+1}", fontsize=10)

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)

plt.show()