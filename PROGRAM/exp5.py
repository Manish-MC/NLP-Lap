import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = input("Enter legal text: ")

# Tokenize and POS tagging
tokens = word_tokenize(text)
tags = pos_tag(tokens)

# Named Entity Recognition
entities = ne_chunk(tags)

print("\nDetected Named Entities:")

count = 0
for entity in entities:
    if hasattr(entity, "label"):
        name = " ".join(word for word, tag in entity.leaves())
        print(f"{name} -> {entity.label()}")
        count += 1

actual = int(input("\nEnter actual number of entities: "))

if max(count, actual) == 0:
    accuracy = 100
else:
    accuracy = (min(count, actual) / max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")