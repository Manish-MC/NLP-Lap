import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Download required resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Experiment title
print("Expt.No: 3")
print("Explore Various Text Similarity Metrics Including")
print("WordNet-Based Similarity for Clustering News Headlines into Topics\n")

# Get headlines
headlines = []

n = int(input("Enter number of headlines: "))

for i in range(n):
    headline = input(f"Enter headline {i+1}: ")
    headlines.append(headline)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(headlines)

# Cosine Similarity
print("\nCosine Similarity Matrix:")
similarity_matrix = cosine_similarity(X)
print(similarity_matrix)

# Clustering
clusters = min(2, n)  # Prevent error if n < 2

kmeans = KMeans(
    n_clusters=clusters,
    random_state=0,
    n_init=10
)

kmeans.fit(X)

print("\nHeadline Clusters:")
for i in range(len(headlines)):
    print(headlines[i], "-> Cluster", kmeans.labels_[i])

# WordNet Similarity
w1 = input("\nEnter first word: ")
w2 = input("Enter second word: ")

s1 = wn.synsets(w1)
s2 = wn.synsets(w2)

if s1 and s2:
    sim = s1[0].path_similarity(s2[0])

    if sim is not None:
        print("WordNet Similarity:", round(sim, 3))
    else:
        print("No similarity score available")
else:
    print("Similarity not found")