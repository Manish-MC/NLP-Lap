import nltk
import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# ----------------------------------------------------
# Download Required NLTK Packages (Run First Time Only)
# ----------------------------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# ----------------------------------------------------
# STEP 1 : Read Unstructured Dataset
# ----------------------------------------------------
with open("customer_reviews.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("="*60)
print("ORIGINAL DATASET")
print("="*60)
print(text)

# ----------------------------------------------------
# STEP 2 : Text Preprocessing
# ----------------------------------------------------

# Convert to Lowercase
text = text.lower()

# Remove Punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokens = word_tokenize(text)

# Remove Stopwords
stop_words = set(stopwords.words("english"))

filtered_words = [
    word for word in tokens
    if word.isalpha() and word not in stop_words
]

print("\n")
print("="*60)
print("PREPROCESSED WORDS")
print("="*60)
print(filtered_words)

# ----------------------------------------------------
# STEP 3 : Text Analysis
# ----------------------------------------------------

print("\n")
print("="*60)
print("TEXT ANALYSIS")
print("="*60)

print("Total Words :", len(filtered_words))
print("Unique Words :", len(set(filtered_words)))

frequency = Counter(filtered_words)

print("\nTop 15 Frequent Words\n")

for word, count in frequency.most_common(15):
    print(word, ":", count)

# ----------------------------------------------------
# STEP 4 : Visualization (Bar Chart)
# ----------------------------------------------------

top_words = frequency.most_common(10)

labels = [x[0] for x in top_words]
values = [x[1] for x in top_words]

plt.figure(figsize=(8,5))
plt.bar(labels, values)
plt.title("Top 10 Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# STEP 5 : Word Cloud
# ----------------------------------------------------

wordcloud = WordCloud(
    width=900,
    height=500,
    background_color="white"
).generate(" ".join(filtered_words))

plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.show()

# ----------------------------------------------------
# STEP 6 : POS Tagging
# ----------------------------------------------------

print("\n")
print("="*60)
print("PART OF SPEECH (POS) TAGGING")
print("="*60)

original_tokens = word_tokenize(text)

pos_tags = nltk.pos_tag(original_tokens)

for word, tag in pos_tags:
    print(f"{word:15} {tag}")

# ----------------------------------------------------
# STEP 7 : Shallow Parsing (Chunking)
# ----------------------------------------------------

print("\n")
print("="*60)
print("SHALLOW PARSING")
print("="*60)

grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
    VP: {<VB.*><NP|PP>*}
"""

chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(pos_tags)

print(tree)

# Opens Parse Tree Window
tree.draw()