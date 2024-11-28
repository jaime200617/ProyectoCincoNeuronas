import nltk
nltk.download('average_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

sentence = "NLTK es una biblioteca de lenguaje natural."
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)
print(pos_tags)