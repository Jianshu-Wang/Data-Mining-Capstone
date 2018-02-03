d1 = "plot: two teen couples go to a church party, drink and then drive."
d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
d4 = "damn that y2k bug . "
documents = [d1, d2, d3, d4]
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
print "tf matrix: "
print tf_matrix

#
# tf_matrixRevised = []
# for x in range(0, len(tf_matrix)):
#     tf_matrixRevised.append([])
#     for y in range(0, 20):
#         tf_matrixRevised[x].append(tf_matrix[x][y])
#
# tf_matrix = tf_matrixRevised
tf_matrix.shape
from sklearn.feature_extraction.text import TfidfTransformer

tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)
print tfidfTran.idf_
import math


def idf(n, df):
    result = math.log((n + 1.0) / (df + 1.0)) + 1
    return result


print "The idf for terms that appear in one document: " + str(idf(4, 1))
print "The idf for terms that appear in two documents: " + str(idf(4, 2))
tfidf_matrix = tfidfTran.transform(tf_matrix)
# print "tfidf_matrix is:"
# print tfidf_matrix
# print "tfidf_matrix array is:"
# arr1 = tf_matrix[0]
# print arr1
#
# arr2 = tf_matrix[0]
# print arr2

import numpy as np
# from sklearn.metrics.pairwise import laplacian_kernel
#
#
# jaccard_similarity_matrix = []
#
# for x in range(0, len(tf_matrix)):
#     jaccard_similarity_matrix.append([])
#     for y in range(0, len(tf_matrix)):
#         result = laplacian_kernel(tf_matrix[x], tf_matrix[y])
#         jaccard_similarity_matrix[x].append(result)
#
#
# print "test result:"
# print jaccard_similarity_matrix
#
#

# value = jaccard_similarity_score(tfidf_matrix.toarray()[1], tfidf_matrix.toarray()[1])
#
# print "test result is"
# print value
