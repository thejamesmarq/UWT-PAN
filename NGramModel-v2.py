from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import argparse, itertools, os
from Util import clean, selectIndexes
from sklearn import svm
from sklearn.pipeline import Pipeline
import xml.etree.ElementTree as ET
from sklearn.feature_selection import SelectKBest, chi2

def run_classifier(train_files,test_files,train_labels,test_labels,gram_type,feature_count=100):

	#Feature extractor
	
	if gram_type == 'comb_grams':
		ngram_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2),token_pattern=r'\b\w+\b',binary=True)
	elif gram_type == 'bi_grams':
		ngram_vectorizer = CountVectorizer(min_df=2, ngram_range=(2, 2),token_pattern=r'\b\w+\b',binary=True)
	elif gram_type == 'uni_grams':
		ngram_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 1),token_pattern=r'\b\w+\b',binary=True)
	else:
		raise Exception("invalid gram type, how did you get that here?")

	#Extract features and fit vocabulary from train set, extract from test set
	train_grams = ngram_vectorizer.fit_transform(train_files).toarray()
	test_grams = ngram_vectorizer.transform(test_files).toarray()

	#Select k best features based on train set, select same features from test set
	ch2 = SelectKBest(chi2, k=feature_count)
	train_grams = ch2.fit_transform(train_grams, train_labels)
	test_grams = ch2.transform(test_grams)

	#Train model (uses Liblinear)
	clf = svm.LinearSVC()
	clf.fit(train_grams, train_labels)

	#Predict values in test set
	predictions = clf.predict(test_grams)

	correct_count = 0

	#Calculate accuracy
	for prediction, actual in zip(test_labels, predictions):
		if prediction == actual:
			correct_count += 1

	return correct_count / float(len(test_labels))

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Evaluates bigram/unigram model')
	parser.add_argument('-i','--input',help='Path to corpus',required=True)
	parser.add_argument('-k','--featurecount',help='Number of features to select',default=100)
	parser.add_argument('-f','--folds',help='Number of folds in cross validation',default=10)
	parser.add_argument('-c','--combinedgrams',help='Use combined grams',default=False)
	parser.add_argument('-b','--bigrams',help='Use bigrams',default=False)
	parser.add_argument('-u','--unigrams',help='Use unigrams',default=False)
	args = parser.parse_args()
	input_path = args.input
	feature_count = int(args.featurecount)
	folds = args.folds

	if not input_path.endswith("/"):
		input_path += "/"

	if args.combinedgrams:
		gram_type = 'comb_grams'
	elif args.bigrams:
		gram_type = 'bi_grams'
	elif args.unigrams:
		gram_type = 'uni_grams'
	else:
		print "Must select an ngram combination (uni,bi,or combined)\nDefaulting to combined"
		gram_type = 'comb_grams'

	truth_in_xml = True
	truth_dic = dict()

	files = [input_path+f for f in os.listdir(input_path) if f.endswith('.xml')]

	labels = []
	ages = []
	genders = []

	texts=[]

	#Get labels either from .xml or truth.txt file
	if os.path.isfile(input_path+"truth.txt"):
		with open(input_path+"truth.txt") as intruth:
			for line in intruth:
				truth_data = line.split(":::")
				if input_path+truth_data[0]+".xml" in files:
					labels.append(truth_data[2]+" "+truth_data[1].lower())
					ages.append(truth_data[2])
					genders.append(truth_data[1].lower())
					truth_dic[truth_data[0]] = (truth_data[2]+" "+truth_data[1].lower(),truth_data[2],truth_data[1])
	else:
		for file in files:
			tree = ET.parse(file)
			root = tree.getroot()
			labels.append(root.attrib["age_group"]+" "+root.attrib["gender"].lower())
			ages.append(root.attrib["age_group"])
			genders.append(root.attrib["gender"].lower())

	tree = ET.parse(files[0])

	#Clean all texts
	for file in files:
		with open(file) as infile:
			texts.append(clean(infile.read(),tree.getroot()))
	
	#Make train/test sets

	comb_skf = StratifiedKFold(labels, folds)
	age_skf = StratifiedKFold(ages, folds)
	gender_skf = StratifiedKFold(genders, folds)

	age_accuracy = []
	gender_accuracy = []
	combined_accuracy = []

	#Train and fit models based on age, gender, and combined labels
	print "Evaluating comb models"
	for train, test in comb_skf:

		train_files = selectIndexes(texts, train)
		test_files = selectIndexes(texts, test)

		train_labels = selectIndexes(labels, train)
		test_labels = selectIndexes(labels, test)

		combined_accuracy.append(run_classifier(train_files,test_files,train_labels,test_labels,gram_type,feature_count))

	#print "All combined accuracies: " + str(combined_accuracy)
	print "Combined accuracy after "+str(folds)+" fold cross validation using "+ gram_type + ": " + str(sum(combined_accuracy)/len(combined_accuracy))

	print "Evaluating age models"
	for train, test in age_skf:
		train_files = selectIndexes(texts, train)
		test_files = selectIndexes(texts, test)

		train_ages = selectIndexes(ages, train)
		test_ages = selectIndexes(ages, test)

		age_accuracy.append(run_classifier(train_files,test_files,train_ages,test_ages,gram_type,feature_count))

	#print "All age accuracies: " + str(age_accuracy)
	print "Age accuracy after "+str(folds)+" fold cross validation using "+ gram_type + ": " + str(sum(age_accuracy)/len(age_accuracy))

	print "Evaluating gender models"
	for train, test in gender_skf:
		train_files = selectIndexes(texts, train)
		test_files = selectIndexes(texts, test)

		train_genders = selectIndexes(genders, train)
		test_genders = selectIndexes(genders, test)

		gender_accuracy.append(run_classifier(train_files,test_files,train_genders,test_genders,gram_type,feature_count))
		
	#print "All gender accuracies: " + str(gender_accuracy)
	print "Gender accuracy after "+str(folds)+" fold cross validation using "+ gram_type + ": " + str(sum(gender_accuracy)/len(gender_accuracy))
	

