# -*- coding: utf-8 -*-

import nltk
import sklearn
import string
import random
import pickle
#import matplotlib

#from sklearn.linear_model import LogisticRegression
#from sklearn.feature_extraction import DictVectorizer
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords



def stemm(wrd):
	stop_words=set(stopwords.words("english"))
	if wrd in stop_words:
		return wrd
	else:
		stemmer=SnowballStemmer("english")
		stem=stemmer.stem(wrd)
		return stem	

def suffix(wrd,stem):
	return(wrd[len(stem):])
	

def creatdict(sentence,index,pos):	#pos=="" <-> featuresofword  else, relative pos (str) is pos
	word=sentence[index]
	wordlow=word.lower()
	stem=stemm(wordlow)
	dict={
		"wrd"+pos:wordlow,
		"stem"+pos:stemm(wordlow),
		"suff"+pos:suffix(wordlow,stem),
		"pref"+pos:wordlow[:3] if len(wordlow)>3 else wordlow[0],
		"cap"+pos:word[0].isupper(),
		"allcap"+pos:word.isupper(),
		"caps_inside"+pos:word==wordlow,
		"nums?"+pos:any(i.isdigit() for i in word),
	}	
	return dict
	

def feature_extractor(sentence,index,postags):
	features=creatdict(sentence,index,"")
	features.update({"postag":postags[index]})

	if index==0:
		 features.update({"first?":True})
	else:
		features.update(creatdict(sentence,index-1,"-1"))
		features.update({"first?":False})	
				
		
	if sentence[-1][-1]!="P": #if no punct at the end
		if index==len(sentence)-1:
			features.update({"last?":True})
		else:
			features.update(creatdict(sentence,index+1,"+1"))
			features.update({"last?":False})	

		
	else:
		if index==len(sentence)-1:
			dummy='1'
		elif index==len(sentence)-2:
			#features.update(creatdict(sentence,index+1))
			features.update({"last?":True})
		else:
			features.update(creatdict(sentence,index+1,"+1"))
			features.update({"last?":False})	

	return features


		
def load(filename):	#filename shall end with .pickle and type(filename)=string
	#print("Loading classifier...")
	with open(filename, "rb") as f:
		classifier=pickle.load(f)
		return classifier


def tag(sentence):
	#one sentence only, as a list
	classifier=load("CRF_ch_all.pickle")
	t_features=[]
	postagged=nltk.pos_tag(sentence)
	sent_postags=[]
	for tup in postagged:
		sent_postags.append(tup[1])
	for j in range(0,len(sentence)):	
		t_features.append(feature_extractor(sentence,j,sent_postags))
		
	#print(sentence)
	#print(len(t_features))	
	
	return classifier.predict([t_features])
