from __future__ import print_function
from gensim.models import KeyedVectors

# Creating the model
en_model = KeyedVectors.load_word2vec_format('/home/m/Data/ft_native_300_ru_wiki_lenta_lemmatize.vec')

# Getting the tokens 
words = []
for word in en_model.vocab:
    words.append(word)

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(words)))

# Printing out the dimension of a word vector 
print("Dimension of a word vector: {}".format(
    len(en_model[words[0]])
))

# Pick a word 
find_similar_to = 'наука'

# Finding out similar words [default= top 10]
for similar_word in en_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))