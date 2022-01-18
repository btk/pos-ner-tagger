# -*- coding: utf-8 -*-

import nltk
import sklearn
import string
import random
import pickle
#import matplotlib
import re

#from sklearn.linear_model import LogisticRegression
#from sklearn.feature_extraction import DictVectorizer
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from do_not_change import tag as chunker
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation



def formatdata(formatted_sentences,formatted_labels,formatted_postags,formatted_chunktags,file_name):
	#file=open("en-ud-dev.conllu","r")
	file=open(file_name, 'r', encoding='ascii', errors='backslashreplace')
	#file=open(file_name,"rb")
	print("Reading data...")
	#quit()
	text=file.read().splitlines()
	tokens=[]
	labels=[]
	postags=[]
	chunktags=[]
	for line in text:
		line=line.split('\t')
		if len(line)==4:
			tokens.append(line[0])
			postags.append(line[1])
			chunktags.append(line[2])
			labels.append(line[3])

		else:
			formatted_sentences.append(tokens)
			formatted_postags.append(postags)
			formatted_chunktags.append(chunktags)
			formatted_labels.append(labels)

			tokens=[]
			labels=[]
			postags=[]
			chunktags=[]




def creatdict(sentence,index,pos,postags,chunktags):	#pos=="" <-> featuresofword  else, relative pos (str) is pos
	word=sentence[index]
	wordhyphen = any(p in word for p in "-")
	wordapos = any(p in word for p in "'")
	wordlength = len(word);
	dict={
		"word"+pos:word.lower(),
		"contains_hyphen"+pos:wordhyphen,
		"contains_apostrophe"+pos:wordapos,
		"capitalized"+pos:word[0].isupper(),
		"all_capital"+pos:word.isupper(),
		"caps_inside"+pos:word==word.lower(),
		"is_alnum"+pos:word.isalnum(),
		"stemmed"+pos:re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word)
	}
	return dict


def feature_extractor(sentence,index,postags,chunktags):
	features=creatdict(sentence,index,"",postags,chunktags)

	return features





def creatsets(file_name):
	sentences=[]
	labels=[] 	#y_train (will be)
	postags=[]
	chunktags=[]
	formatdata(sentences,labels,postags,chunktags,file_name)
	limit=int(len(sentences)/5)###############################***********change these. these just limit the size of training set for faster trials. #####################
	sentences=sentences[:limit]##############
	labels=labels[:limit]####################
	postags=postags[:limit]	#################
	chunktags=chunktags[:limit]	#################
	#print(len(sentences),len(labels),len(postags))
	#print(formatted_sentences)
	#print(formatted_labels)
	print("Feature extraction...")
	features=[]		#X_train
	delimit=int((len(sentences)*8)/10)
	for i in range(0,delimit):
		features.append([])
		for j in range(0,len(sentences[i])):
			features[-1].append(feature_extractor(sentences[i],j,postags[i],chunktags[i]))

	training_data=[features,labels[:delimit]]

	#creating test data (includes pos-retagging)
	t_features=[]
	for i in range(delimit,len(sentences)):
		t_features.append([])
		postagged=nltk.pos_tag(sentences[i])
		chunktagged=chunker(sentences[i])
		sent_postags=[]
		for tup in postagged:
			sent_postags.append(tup[1])
		for j in range(0,len(sentences[i])):
			t_features[-1].append(feature_extractor(sentences[i],j,sent_postags,chunktagged[0]))
		#if i==1:
		#	print(sent_postags)

	test_data=[t_features,labels[delimit:]]


	with open('ner_crf_train.data', 'wb') as file:
		pickle.dump(training_data, file)
	file.close()


	with open('ner_crf_test.data', 'wb') as file:
		pickle.dump(test_data, file)
	file.close()

	return training_data, test_data




def train(training_data):
	print("Training...")
	features=training_data[0]
	labels=training_data[1]
	#print(len(features))
	#print(len(labels))


	for i in range(0,len(features)):
		if len(features[i])!=len(labels[i]):
			del features[i]
			del labels[i]
			i=i-1

	#print(len(features))
	#print(len(labels))

	classifier.fit(features,labels)




def test(test_data):
	print("Testing...")

	features=test_data[0]
	labels=test_data[1]

	#print(len(features))
	#print(len(labels))

	#print(len(features))
	#print(len(labels))

	y_true=labels
	y_pred=classifier.predict(features)

	accuracy=sklearn_crfsuite.metrics.flat_accuracy_score(y_true, y_pred)

	flat_y_true=[]
	flat_y_pred=[]

	for x in y_true:
		for y in x:
			flat_y_true.append(y)

	for x in y_pred:
		for y in x:
			flat_y_pred.append(y)


	#print(type(flat_y_true))
	#print(flat_y_true[0],flat_y_true[-1])

	le=len(flat_y_true)
	i=0
	while i<le:
		if flat_y_true[i][0]=="O" or flat_y_pred[i][0]=="O":
			del(flat_y_true[i])
			del(flat_y_pred[i])
			le=le-1
		else:
			i=i+1


	precision=sklearn.metrics.precision_score(flat_y_true, flat_y_pred,average='micro')
	recall=sklearn.metrics.recall_score(flat_y_true, flat_y_pred,average='micro')
	f1=2*(precision*recall)/(precision+recall)

	print("accuracy:",accuracy)
	print("f1:",f1)
	print("precision:",f1)
	print("recall:",recall)




def save(filename):	#filename shall end with .pickle and type(filename)=string
	print("Saving classifier.")
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return


def load(filename):	#filename shall end with .pickle and type(filename)=string
	print("Loading classifier...")
	with open(filename, "rb") as f:
		classifier=pickle.load(f)
		return classifier


def load(filename):	#filename shall end with .pickle and type(filename)=string
	print("Loading classifier...")
	with open(filename, "rb") as f:
		classifier=pickle.load(f)
		return classifier


def tag(sentence):
	#takes a single sentence as a list
	classifier=load("crf_ner.pickle")
	t_features=[]
	postagged=nltk.pos_tag(sentence)
	chunktagged=chunker(sentence)
	sent_postags=[]
	for tup in postagged:
		sent_postags.append(tup[1])
	for j in range(0,len(sentence)):
		t_features.append(feature_extractor(sentence,j,sent_postags,chunktagged[0]))

	#print(sentence)
	#print(len(t_features))

	ret=classifier.predict([t_features])

	return ret


if __name__ == "__main__":

	classifier=sklearn_crfsuite.CRF(c1=0.2, c2=0.2, max_iterations=1000)
	training_data, test_data=creatsets("eng.train.txt")


	with open('ner_crf_train.data', 'rb') as file:
		training_data=pickle.load(file)
	file.close()


	train(training_data)
	#quit()

	save("crf_ner.pickle")


	with open('ner_crf_test.data', 'rb') as file:
		test_data=pickle.load(file)
	file.close()

	classifier=load("crf_ner.pickle")

	test(test_data)

	s=['The',
	'guitarist',
	'named',
	'Kurt',
	'Cobain',
	'died',
	'of',
	'a',
	'drugs',
	'overdose',
	'in',
	'1970',
	'aged',
	'27',
	'in',
	'London',
	'.']

	print(tag(s))
