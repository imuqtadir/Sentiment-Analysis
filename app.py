import csv
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import json
import string
import math
import numpy as np
import scipy
from scipy.cluster.vq import kmeans,vq
from pylab import *

DEBUG = True

processed_reviews = []

def main():

	reviews = loadFromFile()

	for review in reviews:

		label = review[0]
		text = review[1]

		#if DEBUG:
		print '[' + label + '] = ' + text

		label = label.lower()
		text = text.lower()

		for punc in string.punctuation:
			text = text.replace(punc, '')

		# Process text
		words = process(label, text)
		createFreqDistribution(label, words)

	words = computeFreqDistribution()

	no_of_words = len(words)
	no_of_reviews = len(reviews)

	feature_vector = [[0 for i in range(no_of_words)] for j in range(no_of_reviews)]

	if DEBUG:
		printArray(feature_vector)

	r = 0
	for label, review_words in processed_reviews:
		c = 0
		for word in words:
			if word in review_words:
				feature_vector[r][c] = 1
			else:
				feature_vector[r][c] = 0
			c += 1
		r += 1

	if DEBUG:
		printArray(feature_vector)
         

	similarity_matrix = computeCosineSimilarity(feature_vector)
	print "SIMILARITY MATRIX:"
	#if DEBUG:
	printArray(similarity_matrix)
	
	temp,l=computeDiagLap(similarity_matrix)
	print "STEP 1:"
	print "DIAGONAL MATRIX:"
	printArray(temp)
	print "LAPLACIAN MATRIX:"
	printArray(l)

	vals,vecs,lval,lvec=computeEig(l)
	print "STEP 2:"
	print "EIGEN VALUES & EIGEN VECTORS:"
	printArray(vals)
	printArray(vecs)
	print "STEP 3:"
	print "LARGEST 2 EIGEN VALUES & EIGEN VECTORS:"
	printArray(lval)
	printArray(lvec)

	norm=computeNormalized(lvec)
	print "STEP 4:"
	print "NORMALIZED MATRIX:"
	printArray(norm)
	print "STEP 5:"
	kmeans2clus(norm)
	
def loadFromFile():

	temp = []

	with open('book.csv', 'rb') as file:
		reader = csv.reader(file, delimiter = ',', quotechar = '"')

		for row in reader:
			temp.append(row)

	return temp

def process(label, text):

	# Split text to words
	words = text.split(' ')

	words = removeStopWords(words)

	words = removeDomainWords(words)

	words = applyStemmer(words)

	processed_reviews.append([label, words])

	return words

def removeStopWords(words):

	if DEBUG:
		print 'In Remove Stop Words'

	temp = []

	# Check the output BEFORE removing
	if DEBUG:
		print words

	# Remove stop words
	stopset = set(stopwords.words('english'))
	for word in words:
		if word not in stopset:
			if len(word) > 1:
				temp.append(word)

	words = temp

	# Check the output AFTER removing
	if DEBUG:
		print words

	return words

def removeDomainWords(words):

	if DEBUG:
		print 'In Remove Domain Words'

	domainWords = ['book','readers','reading','read','story','plot','author'] # Add more domain names here

	temp = []

	# Check the output BEFORE removing
	if DEBUG:
		print words

	for word in words:
		if word not in domainWords:
			temp.append(word)

	words = temp

	# Check the output AFTER removing
	if DEBUG:
		print words

	return words

def applyStemmer(words):

	if DEBUG:
		print 'In Apply Stemmer'

	lmtzr = WordNetLemmatizer()

	temp = []

	# Check the output BEFORE stemming
	if DEBUG:
		print words

	for word in words:
		temp.append(lmtzr.lemmatize(word))

	words = temp

	# Check the output AFTER stemming
	if DEBUG:
		print words

	return words

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()
def createFreqDistribution(label, words):

	if DEBUG:
		print 'In Create Freq Distribution'

	for word in words:
		word_fd.inc(word)
		label_word_fd[label].inc(word)

	if DEBUG:
		print word_fd

def computeFreqDistribution():

	if DEBUG:
		print word_fd

	pos_word_count = label_word_fd['positive'].N()
	neg_word_count = label_word_fd['negative'].N()
	neu_word_count = label_word_fd['neutral'].N()
	total_word_count = pos_word_count + neg_word_count + neu_word_count

	word_scores = {}

	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(label_word_fd['positive'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(label_word_fd['negative'][word], (freq, neg_word_count), total_word_count)
		neu_score = BigramAssocMeasures.chi_sq(label_word_fd['neutral'][word], (freq, neu_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score + neu_score

	if DEBUG:
		print json.dumps(word_scores, indent = 4)

	threshold = 2

	temp = []

	for item in word_scores:
		if word_scores[item] > threshold:
			temp.append(item)

	if DEBUG:
     
		print temp
                
	return temp


def printArray(input):
	for item in input:
		print item

def computeCosineSimilarity(feature_vector):

	temp = [[0 for i in range(len(feature_vector))] for j in range(len(feature_vector))]

	for i in range(len(feature_vector)):
		for j in range(len(feature_vector)):
			if i < j:
				#print i, j
				d1 = feature_vector[i]
				d2 = feature_vector[j]

				xy = 0
				sqx = 0
				sqy = 0

				for k in range(len(d1)):
					xy += d1[k] * d2[k]
					sqx += d1[k]**2
					sqy += d2[k]**2

				temp[i][j] = temp[j][i] = xy / (math.sqrt(sqx) * math.sqrt(sqy))
				#print d1, d2

	return temp

def computeDiagLap(similarity_matrix):
        temp =[[0 for i in range(len(similarity_matrix))] for j in range(len(similarity_matrix))]
        for i in range(len(similarity_matrix)):
                for j in range(len(similarity_matrix)):
                        summ=sum(similarity_matrix[i])
                        temp[i][i]=summ
        temp=np.matrix(temp)
        c=np.linalg.matrix_power(temp,-(1/2))
        a=np.matrix(similarity_matrix)
        e=c*a
        l=e*c
        return temp,l



def computeEig(l):
        vals,vecs=np.linalg.eig(l)
        return vals,vecs,vals[0:2],vecs[:,0:2]

def computeNormalized(lvec):
        norm=scipy.cluster.vq.whiten(lvec)
        return norm


def kmeans2clus(norm):
        centroids,_ = kmeans(norm,3)
        
        idx,_ = vq(norm,centroids)


        plot(norm[idx==0,0],norm[idx==0,1],'ob',
             norm[idx==1,0],norm[idx==1,1],'or',
             norm[idx==2,0],norm[idx==2,1],'og')
        plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
        show()

# now with K = 3 (3 clusters)
        #centroids,_ = kmeans(norm,3)
        #idx,_ = vq(norm,centroids)

        #plot(norm[idx==0,0],norm[idx==0,1],'ob',
        #norm[idx==1,0],norm[idx==1,1],'or',
        #norm[idx==2,0],norm[idx==2,1],'og') # third cluster points
        #plot(centroids[:,0],centroids[:,1],'sm',markersize=8)
        #show()

# Program starts here...
if __name__ == '__main__':

	main()
