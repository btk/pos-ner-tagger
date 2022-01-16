# -*- coding: utf-8 -*-

import nltk
import sklearn
import string
import random
import pickle
import numpy

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from do_not_change import tag as chunker
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords



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
	wordlow=word.lower()
	dict={
		"wrd"+pos:wordlow,
		"cap"+pos:word[0].isupper(),
		"allcap"+pos:word.isupper(),
		"caps_inside"+pos:word==wordlow,
		"nums?"+pos:any(i.isdigit() for i in word),
	}	
	return dict
	

def feature_extractor(sentence,index,postags,chunktags):
	features=creatdict(sentence,index,"",postags,chunktags)

	return features


			
def creatsets(file_name):	
	sentences=[]
	s_labels=[] 	#y_train (will be)
	postags=[]
	chunktags=[]
	formatdata(sentences,s_labels,postags,chunktags,file_name)
	#print(len(sentences),len(s_labels),len(postags))
	limit=int(len(sentences)/4)################ THIS PART LIMITS THE LENGTH OF THE TRAINING SET FOR FAST TRIALS. YOU MAY (AND AT THE END, HAVE TO) CHANGE THIS ###
	sentences=sentences[:limit]##############																													 #
	s_labels=s_labels[:limit]################																													 #
	postags=postags[:limit]	#################																													 #
	chunktags=chunktags[:limit]	###############				###					###						###						###				########	   ###
	#print(len(sentences),len(s_labels),len(postags))			
	#print(formatted_sentences)
	#print(formatted_labels)
	print("Feature extraction...")
	features=[]		#X_train
	labels=[]
	delimit=int((len(sentences)*8)/10)
	for i in range(0,delimit):
		for j in range(0,len(sentences[i])):
			features.append(feature_extractor(sentences[i],j,postags[i],chunktags[i]))
			labels.append(s_labels[i][j])
	
	#print("postags0",postags[0])
	
	"""
	print(postags[0])
	print(training_data[0][0])
	print(len(training_data[0]))
	print(len(training_data[0][0]))
	print("\n")
	print(training_data[1][0])
	print(len(training_data[1]))
	print(len(training_data[1][0]))
	"""	
	#creating test data (includes pos-retagging)		
	t_features=[]
	t_labels=[]
	for i in range(delimit,len(sentences)):		
		postagged=nltk.pos_tag(sentences[i])
		chunktagged=chunker(sentences[i])
		sent_postags=[]
		for tup in postagged:
			sent_postags.append(tup[1])
		for j in range(0,len(sentences[i])):
			t_features.append(feature_extractor(sentences[i],j,sent_postags,chunktagged[0]))
			t_labels.append(s_labels[i][j])
		#if i==delimit:
		#	print("sent_postags",sent_postags)	
			
			
	#print(t_labels[:2])		
	
	
	"""
	print(test_data[0][0])
	print(len(test_data[0]))
	print(len(test_data[0][0]))
	print("\n")
	print(test_data[1][0])
	print(len(test_data[1]))
	print(len(test_data[1][0]))
	"""
			
	
	del sentences[:]
	del sentences
	del s_labels[:]
	del s_labels
	
	
	"""
	print(len(features))
	print(len(labels))
	print(features[0])
	print(labels[0])
	"""
	
	print("Vectorizing...")
	vectorizer=DictVectorizer()
	visa=vectorizer.fit(features)
	v_ized=visa.transform(features)
	
	"""
	print(len(labels))
	print(v_ized.shape[0], v_ized.shape[1])	

	print(v_ized[0])
	#print(vectorizer.inverse_transform(v_ized)[0])
	"""
	training_data=[v_ized,labels]
	


	t_v_ized=visa.transform(t_features)
	test_data=[t_v_ized,t_labels]

	
	#quit()
	with open('ner_lr_vectorizer.pickle', 'wb') as file:
		pickle.dump(visa, file)
	file.close()	
	
	
	with open('ner_lr_train.data', 'wb') as file:
		pickle.dump(training_data, file)
	file.close()


	with open('ner_lr_test.data', 'wb') as file:
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

	features=test_data[0]
	labels=test_data[1]


	y_true=labels  
	y_pred=classifier.predict(features)
	#print(y_true[:2])
	#print(y_pred[:2])
	
	accuracy=sklearn.metrics.accuracy_score(y_true, y_pred)

	flat_y_pred=[]
	for x in y_pred:
			flat_y_pred.append(x)
			
			
	#print(len(y_true))		
	#print(len(y_pred))
	#print(len(flat_y_pred))
	#print(type(y_true),type(flat_y_pred))
	#print(len(labels))
	le=len(labels)
	i=0
	while i<le:
		if y_true[i][0]=="O" or flat_y_pred[i][0]=="O":
			del(y_true[i])
			del(flat_y_pred[i]) 
			le=le-1
		else:
			i=i+1
			
					
	precision=sklearn.metrics.precision_score(y_true, flat_y_pred,average='micro')
	recall=sklearn.metrics.recall_score(y_true, flat_y_pred,average='micro')
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


def load_vectorizer(filename):	#filename shall end with .pickle and type(filename)=string
	with open(filename, "rb") as f:
		vect_shaper=pickle.load(f)
		return vect_shaper

def tag(sentence):		# tags given sentence, for demonstration purposes.
	vectorizer=load_vectorizer('ner_lr_vectorizer.pickle')
	classifier=load("ner_lr.pickle")
	postagged=nltk.pos_tag(sentence)
	chunktagged=chunker(sentence)
	sent_postags=[]
	for tup in postagged:
		sent_postags.append(tup[1])
	t_features=[]
	for j in range(0,len(sentence)):	
		t_features.append(feature_extractor(sentence,j,sent_postags,chunktagged[0]))
		
	ret=classifier.predict(vectorizer.transform(t_features))	
		
	return ret
	
	

if __name__ =="__main__":
	
	classifier=LogisticRegression(max_iter=1000 , multi_class='multinomial')
	training_data, test_data=creatsets("eng.train.txt")
	
	with open('ner_lr_train.data', 'rb') as file:
		training_data=pickle.load(file)
	
	
	train(training_data)
	#quit()
	
	save("ner_lr.pickle")
	
	
	with open('ner_lr_test.data', 'rb') as file:
		test_data=pickle.load(file)
	
	classifier=load("ner_lr.pickle")
	
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