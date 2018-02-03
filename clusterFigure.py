from numpy import genfromtxt

my_data = genfromtxt('tfidfResult.csv', delimiter=',')

# my_data = [[0], [3], [1]]

from sklearn.cluster import SpectralClustering


def spectral(k, D, rs):
    """
    From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    # if D[1, 1] < 1: D = 1 - D  # Convert distance to similarity matrix
    spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=rs)
    spectral.fit(D)
    labels = spectral.labels_
    return labels

k = 5
label = spectral(k, my_data, None)
labelStr = ''
print label
header = "Burgers,French,Soul Food,Fondue,Soup,Do-It-Yourself Food,Ukrainian,Middle Eastern,Coffee & Tea,Wine Bars,Russian,Brazilian,Barbeque,Live-Raw Food,Indonesian,Fast Food,Seafood,Ramen,Hawaiian,Latin American,Shanghainese,Juice Bars & Smoothies,Persian-Iranian,Kosher,Sandwiches,Tea Rooms,Chicken Wings,Donuts,Cocktail Bars,Desserts"
headerArray = header.split(",")
strList = []
for I in range(0,k):
    for index in range(0,len(label)):
        if label[index] == I:
            strList.append(headerArray[index])
            labelStr = labelStr + ',' + str(label[index])


print labelStr
print strList
strRes = ''
for i in range(0,len(strList)):
    strRes = strRes + ',' + strList[i]

print strRes

f = open('clusters.txt','w')
f.write(labelStr)
f.write('\n'+strRes)
f.close()






# from sklearn.neighbors import kneighbors_graph
# A = kneighbors_graph(my_data, 6, mode='connectivity', include_self=True)
# print A.toarray()


# import csv
#
# with open("tfidfResult_clusters.csv", "w+") as my_csv:
#     csvWriter = csv.writer(my_csv, delimiter=',')
#     csvWriter.writerows(label)
