# d1 = "plot: two teen couples go to a church party, drink and then drive."
# d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
# d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
# d4 = "damn that y2k bug . "
# documents = [d1, d2, d3, d4]

fileName = "Burgers,French,Soul Food,Fondue,Soup,Do It Yourself Food,Ukrainian,Middle Eastern,Coffee & Tea,Wine Bars,Russian,Brazilian,Barbeque,Live Raw Food,Indonesian,Fast Food,Seafood,Ramen,Hawaiian,Latin American,Shanghainese,Juice Bars & Smoothies,Persian Iranian,Kosher,Sandwiches,Tea Rooms,Chicken Wings,Donuts,Cocktail Bars,Desserts"
fileNameList = fileName.split(",")

documents = []
path = 'Categories/'

for name in fileNameList:
    currentPath = path + name + '.txt'
    with open(currentPath) as f:
        data = f.read()  # .replace('\n', '')
        documents.append(data)

import nltk, string, numpy

nltk.download('punkt')  # first-time use only
stemmer = nltk.stem.porter.PorterStemmer()


def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


nltk.download('wordnet')  # first-time use only
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


from sklearn.feature_extraction.text import CountVectorizer

LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(documents)
print LemVectorizer.vocabulary_
tf_matrix = LemVectorizer.transform(documents).toarray()
print tf_matrix


import numpy as np
from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_matrix = []

for x in range(0, len(tf_matrix)):
    jaccard_similarity_matrix.append([])
    for y in range(0, len(tf_matrix)):
        result = jaccard_similarity_score(tf_matrix[x], tf_matrix[y])
        jaccard_similarity_matrix[x].append(result)


print "test result:"
print jaccard_similarity_matrix

import csv

with open("tfResult2_jaccard.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(jaccard_similarity_matrix)



