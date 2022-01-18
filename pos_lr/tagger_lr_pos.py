# -*- coding: utf-8 -*-

import nltk
import sklearn
import string
import random
import pickle
import matplotlib
import numpy

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer



def formatdata(formatted_sentences,formatted_labels,file_name):
	file=open(file_name, 'r', encoding='utf-8', errors='backslashreplace')
	print("Reading data...")
	text=file.read().splitlines()
	tokens=[]
	labels=[]
	for line in text:
		line=line.split('\t')
		if len(line)==3:
			tokens.append(line[0])
			if line[1]=="PUNCT":
				labels.append(line[0]+"P")		# identifies puntuations with P, without losing information. (i.e. "," is still comma but in the form of ".P", thus distinguishable from ".P")
			else:
				labels.append(line[2])
		else:									#for empty line (i.e end of sentence)
			formatted_sentences.append(tokens)
			formatted_labels.append(labels)
			tokens=[]
			labels=[]




def creatdict(sentence,index,pos):	#pos=="" for features of tokens;  else relative pos (str) is the value of the pos variable.
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
	s_labels=[]
	formatdata(sentences,s_labels,file_name)
	limit=int(len(sentences)/5)###############################********************#####################
	sentences=sentences[:limit]##############
	s_labels=s_labels[:limit]####################

	#print(len(sentences),len(s_labels))

	print("Feature extraction...")
	delimit=int((len(sentences)*8)/10)

	features=[]		#X_train
	labels=[]		#Y_train
	for i in range(0,delimit):
		for j in range(0,len(sentences[i])):
			features.append(feature_extractor(sentences[i],j))
			labels.append(s_labels[i][j])


	t_features=[]
	t_labels=[]
	for i in range(delimit,len(sentences)):
		for j in range(0,len(sentences[i])):
			t_features.append(feature_extractor(sentences[i],j))
			t_labels.append(s_labels[i][j])


	del sentences[:]
	del s_labels[:]

	del sentences
	del s_labels


	print("Vectorizing...")
	vectorizer=DictVectorizer()
	visa=vectorizer.fit(features)
	v_ized=visa.transform(features)

	training_data=[v_ized,labels]


	t_v_ized=visa.transform(t_features)
	test_data=[t_v_ized,t_labels]


	with open('pos_lr_vectorizer.pickle', 'wb') as file:
		pickle.dump(visa, file)

	with open('pos_lr_train.data', 'wb') as file:
		pickle.dump(training_data, file)

	with open('pos_lr_test.data', 'wb') as file:
		pickle.dump(test_data, file)

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

	#print(y_pred[0])

	end_p=["RP","NFP","VBP","NNP","PRP","WP"]
	for i in range(0,len(labels)):
		if y_true[i][-1]=="P" and y_true[i][-1] not in end_p:
			y_true[i]="PUNCT"
		if y_pred[i][-1]=="P" and y_pred[i][-1] not in end_p:
			y_pred[i]="PUNCT"


	precision=sklearn.metrics.precision_score(y_true, y_pred,average='micro')
	recall=sklearn.metrics.recall_score(y_true, y_pred,average='micro')
	f1=2*(precision*recall)/(precision+recall)
	accuracy=sklearn.metrics.accuracy_score(y_true, y_pred)
	print("accuracy:",accuracy)
	print("f1:",f1)
	print("precision:",precision)
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

def tag(sentence):
	vectorizer=load_vectorizer('pos_lr_vectorizer.pickle')
	classifier=load("pos_lr.pickle")

	t_features=[]
	for j in range(0,len(sentence)):
		t_features.append(feature_extractor(sentence,j))

	ret=classifier.predict(vectorizer.transform(t_features))

	end_p=["RP","NFP","VBP","NNP","PRP","WP"]
	for i in range(0,len(ret)):
		if ret[i][-1]=="P" and ret[i][-1] not in end_p:
			ret[i]="PUNCT"

	return ret


if __name__ =="__main__":

	classifier=LogisticRegression(max_iter=1000,multi_class='multinomial')

	training_data, test_data=creatsets("en-ud-train.conllu")

	with open('pos_lr_train.data', 'rb') as file:
		training_data=pickle.load(file)
	file.close()

	train(training_data)

	save("pos_lr.pickle")

	with open('pos_lr_test.data', 'rb') as file:
		test_data=pickle.load(file)
	file.close()


	classifier=load("pos_lr.pickle")

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
