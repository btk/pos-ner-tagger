# -*- coding: utf-8 -*-

import nltk
import sklearn
import string
import random
import pickle
import numpy

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


def formatdata(formatted_sentences,formatted_labels,file_name):
	#file=open("en-ud-dev.conllu","r")
	file=open(file_name, 'r', encoding='ascii', errors='backslashreplace')
	#file=open(file_name,"rb")
	print("Reading data...")
	#quit()
	text=file.read().splitlines()
	tokens=[]
	labels=[]
	for line in text:
		line=line.split('\t')
		if len(line)==3:
			tokens.append(line[0])
			if line[1]=="PUNCT":
				labels.append(line[0]+"P")
			else:
				labels.append(line[2])	
		else:
			formatted_sentences.append(tokens)
			formatted_labels.append(labels)
			tokens=[]
			labels=[]



	

def creatdict(sentence,index,pos):	#pos=="" <-> featuresofword  else, relative pos (str) is pos
	word=sentence[index]
	wordlow=word.lower()
	dict={
		"wrd"+pos:wordlow,								# the token itself
		"cap"+pos:word[0].isupper(),					# starts with capital?
		"allcap"+pos:word.isupper(),					# is all capitals?
		"caps_inside"+pos:word==wordlow,				# has capitals inside?
		"nums?"+pos:any(i.isdigit() for i in word),		# has digits?
	}	
	return dict
	

def feature_extractor(sentence,index):
	features=creatdict(sentence,index,"")

	return features



			
def creatsets(file_name):	
	sentences=[]
	labels=[] 	#y_train (will be)
	formatdata(sentences,labels,file_name)	
	limit=int(len(sentences)/5)##############
	sentences=sentences[:limit]##############
	labels=labels[:limit]####################
	
	#print(len(sentences),len(labels))			
	#print(formatted_sentences)
	#print(formatted_labels)
	print("Feature extraction...")
	features=[]		#X_train
	for i in range(0,len(sentences)):
		features.append([])
		for j in range(0,len(sentences[i])):
			features[-1].append(feature_extractor(sentences[i],j))
			
	del sentences[:]
	del sentences

	
	delimit=int((len(labels)*8)/10)
	test_data=[features[delimit:],labels[delimit:]]
	features=features[:delimit]
	labels=labels[:delimit]
	
	training_data=[features,labels]

	
	with open('pos_crf_train.data', 'wb') as file:
		pickle.dump(training_data, file)
	file.close()


	with open('pos_crf_test.data', 'wb') as file:
		pickle.dump(test_data, file)
	file.close()
		
	return training_data, test_data	
	
	
		
def train(training_data):		
	print("Training...")
	features=training_data[0]
	labels=training_data[1]	
	classifier.fit(features,labels)	
	



def test(test_data):
	print("Testing...")

	y_true=test_data[1]  #labels
	y_pred=classifier.predict(test_data[0])
	
	#print(y_pred[0])
	
	precision=sklearn_crfsuite.metrics.flat_precision_score(y_true, y_pred,average='micro')
	recall=sklearn_crfsuite.metrics.flat_recall_score(y_true, y_pred,average='micro')
	f1=2*(precision*recall)/(precision+recall)
	accuracy=sklearn_crfsuite.metrics.flat_accuracy_score(y_true, y_pred)

	print("accuracy:",accuracy)
	print("f1:",f1)
	print("precision:",f1)
	print("recall:",recall)
	
	
	import plotly
	import plotly.graph_objects as go

	flat_y_true=[]
	flat_y_pred=[]
	
	for x in y_true:
		for y in x:
			flat_y_true.append(y)
	
	for x in y_pred:
		for y in x:
			flat_y_pred.append(y)		
	
	end_p=["RP","NFP","VBP","NNP","PRP","WP"]
	for i in range(0,len(flat_y_true)):
		if flat_y_true[i][-1]=="P" and flat_y_true[i][-1] not in end_p: 
			flat_y_true[i]="PUNCT"
		if flat_y_pred[i][-1]=="P" and flat_y_pred[i][-1] not in end_p: 
			flat_y_pred[i]="PUNCT"
		
	#print(type(flat_y_true))
	#print(flat_y_true[0],flat_y_true[-1])	

	

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




def tag(sentence):
	#takes a single sentence as a list
	classifier=load("pos_crf.pickle")
	t_features=[]
	for j in range(0,len(sentence)):	
		t_features.append(feature_extractor(sentence,j))
		
	#print(sentence)
	#print(len(t_features))	
	
	ret=classifier.predict([t_features])[0]
	end_p=["RP","NFP","VBP","NNP","PRP","WP"]
	for i in range(0,len(ret)):
		if ret[i][-1]=="P" and ret[i][-1] not in end_p: 
			ret[i]="PUNCT"

	return ret



if __name__ == "__main__":

	classifier=sklearn_crfsuite.CRF(c1=0.2, c2=0.2, max_iterations=1000)
	training_data, test_data=creatsets("en-ud-train.conllu")
	
	
	with open('pos_crf_train.data', 'rb') as file:
		training_data=pickle.load(file)
	file.close()
	
	
	train(training_data)
	#quit()
	save("pos_crf.pickle")
	
	
	with open('pos_crf_test.data', 'rb') as file:
		test_data=pickle.load(file)
	file.close()
	
	classifier=load("pos_crf.pickle")
	test(test_data)
	
	s=['The',
	'guitarist',
	'died',
	'of',
	'a',
	'drugs',
	'overdose',
	'in',
	'1970',
	'aged',
	'27',
	'.']
	
	print(tag(s))