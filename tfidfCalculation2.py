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
        data = f.read().replace('\n', '')
        documents.append(data)



#
# import glob
#
# path = 'Categories/*.txt'
# files = glob.glob(path)
#
# for name in files:
#     try:
#         with open(name) as f:
#             data = f.read().replace('\n', '')
#             documents.append(data)
#     except IOError as exc:
#         if exc.errno != errno.EISDIR:
#             raise


# with open('data.txt', 'r') as myfile:
#     data=myfile.read().replace('\n', '')

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
tf_matrix = tf_matrix*10000
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
print "The tfidf matrix is: "
print tfidf_matrix.toarray()
cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print "The cosine similarity matrix is: "
print cos_similarity_matrix


import csv

with open("tfidfResult2.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(cos_similarity_matrix)
