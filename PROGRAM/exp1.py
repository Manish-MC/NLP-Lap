import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample input sentences
samples = [
    "The boys are playing football in the playground.",
    "Cats were chasing mice around the house.",
    "She studies machine learning every day.",
    "The leaves are falling from the trees.",
    "Running is healthier than sitting all day."
]

print("Choose an input sentence:")
for i, sentence in enumerate(samples, start=1):
    print(f"{i}. {sentence}")
print("6. Enter your own sentence")

choice = int(input("\nEnter your choice (1-6): "))

if choice == 6:
    text = input("Enter a sentence: ")
elif 1 <= choice <= 5:
    text = samples[choice - 1]
else:
    print("Invalid choice! Using default sentence.")
    text = samples[0]

# Tokenization
tokens = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Display Results
print("\n==============================")
print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nStemmed Words:")
print(stemmed_words)

print("\nLemmatized Words:")
print(lemmatized_words)

print("\nComparison:")
print("- Stemming removes prefixes/suffixes to produce root forms.")
print("- Lemmatization converts words into meaningful dictionary base forms.")
print("==============================")