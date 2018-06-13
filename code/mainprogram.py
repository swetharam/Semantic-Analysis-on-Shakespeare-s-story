import nltk
import numpy as np
import gensim
from sklearn.cluster import KMeans
import re
import nltk
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn.decomposition import PCA

stop_words =set(stopwords.words('english'))
filtered_sent=[]
def removestops(list):
    for lists in list:
        temp=[]
        for words in lists:
            if words not in stop_words:
                temp.append(words)
        filtered_sent.append(temp)
sent2=[]
def wordclean(flist):
    for lists in flist:
        temp = []
        for word in lists:
            if word=="." or word=="""""" or word=="." or word=="'" or word=="``" or word==",":
                continue
            else:
                temp.append((word))
            sent2.append(temp)

with open("document.txt", 'r')as in_file:
    text = in_file.read()
    sents = nltk.sent_tokenize(text)
print(sents)
list=[]
for sent in sents:
    list.append(nltk.word_tokenize(sent))

removestops(list)
wordclean(filtered_sent)
print(sent2)

model=gensim.models.Word2Vec(sent2,min_count=10,workers=4,window=5,hs=1,negative=0)
# words=model.score("the waiter and old man were sad".split())
# print(words)



# a=model.score(["The old man was depressed".split()])
# print(a)
# print(model)
words = model.wv.vocab
# model.wv.save_word2vec_format('model.txt',binary=False)
# print(model.wv.accuracy())n b
# print(words)
# print(model.wv.most_)
# f1 = model[model.wv.vocab]
X = model[model.wv.vocab]
pca = PCA(n_components=3)
result = pca.fit_transform(X)
# create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
words = model.wv.vocab
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
# X1=np.matrix(f1)
# kmeans = KMeans(n_clusters=2,random_state=0).fit(result)
# print(words)
# centroids = kmeans.cluster_centers_
# pyplot.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# print(kmeans.labels_)
# temp=[]
# for word in words:
#     temp.append(word)
#
# print(temp)
# # pyplot.show()
