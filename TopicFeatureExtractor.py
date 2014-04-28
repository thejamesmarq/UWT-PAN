import gensim
from gensim.utils import tokenize
from gensim import models
from gensim.corpora.textcorpus import TextCorpus
import os, string, csv, numpy,gc,itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from pylab import *
from sys import getsizeof
#from nltk import clean_html
#from nltk import bigrams
from nltk.tokenize import word_tokenize
from sklearn.feature_selection import SelectKBest, chi2
from nltk.stem.snowball import SnowballStemmer
from scipy import dot, linalg, mat
import xml.etree.ElementTree as ET
import os
from Util import clean
from scipy.stats import pearsonr

class TopicModels:

	def __init__(self, input_path):
		self.topics = 100
		files = [input_path+f for f in os.listdir(input_path) if f.endswith('.xml')]
		mycorpus = MyCorpus(files)

		self.seen_documents = []
		for document in mycorpus:
			self.seen_documents.append(document)

		self.lda_model = models.LdaModel(mycorpus, num_topics = self.topics, eval_every = 5)

		group_corpora = []
		self.group_topics = []
		for label, list in self.make_doc_dict(files).iteritems():
			group_corpora.append(MyCombCorpus((list,files)))

		for corp in group_corpora:
			for doc in corp:
				vector = [0]*self.topics
				this_topics = self.lda_model[doc]
				for tuple in this_topics:
					vector[tuple[0]]=tuple[1]
				self.group_topics.append(vector)

	def get_sim_seen(self, index):
		ret_sims = []
		p = self.lda_model[self.seen_documents[index]]
		vector = [0]*self.topics
		for tuple in p:
			vector[tuple[0]]=tuple[1]
		p = vector
		
		for q in self.group_topics:
			'''
			sim = numpy.dot(p, q) / (numpy.sqrt(numpy.dot(p, p)) * numpy.sqrt(numpy.dot(q, q)))
			ret_sims.append(sim)
			'''
			'''
			p = np.asarray(p, dtype=np.float)
    		q = np.asarray(q, dtype=np.float)
 			
    		ret_sims.append(np.sum(np.where(p != 0, p * np.log(p / q), 0)))
			'''
			ret_sims.append(pearsonr(p,q)[0])

		return ret_sims

	def get_sim_unseen(self, file):
		ret_sims = []
		mycorpus = MyCorpus([file])
		document = []
		for doc in mycorpus:
			document.append(doc)
		p = self.lda_model[document[0]]
		vector = [0]*self.topics
		for tuple in p:
			vector[tuple[0]]=tuple[1]
		p = vector
		for q in self.group_topics:
			'''
			sim = numpy.dot(p, q) / (numpy.sqrt(numpy.dot(p, p)) * numpy.sqrt(numpy.dot(q, q)))
			ret_sims.append(sim)
			'''
			ret_sims.append(pearsonr(p,q)[0])
			
		return ret_sims

	def make_doc_dict(self, files):
		doc_labels = dict()

		index = 0
		for file in files:
			tree = None
			try:
				tree = ET.parse(file)
			except ET.ParseError:
				index += 1
				continue
			root = tree.getroot()
			age = root.get('age_group')
			gender = root.get('gender')
			label = age + " " + gender

			if not label in doc_labels:
				doc_labels[label] = []
			if not age in doc_labels:
				doc_labels[age] = []
			if not gender in doc_labels:
				doc_labels[gender] = []

			doc_labels[label].append(index)
			doc_labels[age].append(index)
			doc_labels[gender].append(index)
			index+=1

		return doc_labels

class MyCorpus(gensim.corpora.TextCorpus): 

	def get_texts(self): 
		for filename in self.input:
			root = ET.fromstring(open(filename).read())
			lang = root.attrib['lang'].lower()
			genre = root.attrib['type']
			tree = ET.ElementTree(root)
			yield tokenize(clean(open(filename).read(),lang,genre,tree))

class MyCombCorpus(gensim.corpora.TextCorpus): 

	def get_texts(self): 
		text = ""
		for index in self.input[0]:
			root = ET.fromstring(open(self.input[1][index]).read())
			lang = root.attrib['lang'].lower()
			genre = root.attrib['type']
			tree = ET.ElementTree(root)
			string = clean(open(self.input[1][index]).read(),lang,genre,tree)
			text += string
		yield tokenize(text)