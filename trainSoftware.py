import argparse, os, csv, string, pickle, numpy
import xml.etree.ElementTree as ET
from sklearn import svm
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.tag.stanford import POSTagger
from nltk.corpus import cmudict
from LIWC import extractLIWCFeatures, readDictionary, getLIWCOnLang
from Util import removeStopWords, extractLIWCFeatures, extractRepeatation, extractCapital, clean, writeCSV, extractEmoticons, extractHTMLTags
from TopicFeatureExtractor import TopicModels
from ReadabilityCalculator import extractReadabilityArray
from sklearn import preprocessing

def run(args):
	#print "test, fix these before deployment"
	#input_path = "/Users/jamarq_laptop/PAN/Software/pan-2/Data/blogs/es/"
	#out_path = "./Models/"

    input_path = args.input
    output_path = args.output

    if not input_path.endswith("/"):
        input_path += "/"
    if not output_path.endswith("/"):
        output_path += "/"

    truth_in_xml = True
    truth_dic = dict()

    if os.path.isfile(input_path+"truth.txt"):
        print "using truth.txt file"
        truth_in_xml = False
        with open(input_path+"truth.txt","rb") as truth_in:
            for line in truth_in:
                truth_data = line.split(":::")
                truth_dic[truth_data[0]] = (truth_data[2],truth_data[1])

    print "input path: " + input_path
    topic_model = TopicModels(input_path)
    #print example.get_divergences_seen(1)

    files = [f for f in os.listdir(input_path) if f.endswith('.xml')]
    print str(len(files)) + " files in corpus"

    tree = ET.parse(input_path + files[0])
    root = tree.getroot()

    type = root.attrib["type"] + "_" + root.attrib["lang"]

    features = []
    labels = []
    ages = []
    genders = []
    ids = []

    doc_index = 0
    for file in files:

    	these_features = []

        these_features.extend(topic_model.get_sim_seen(doc_index))

    	tree = ET.parse(input_path + file)
    	root = tree.getroot()
    	these_features.append(int(tree.find("documents").attrib["count"]))
    	
    	xmlstr = clean(ET.tostring(root),root.attrib['lang'].lower(),root.attrib['type'],tree)
    	xmlstr = filter(lambda x: x in string.printable, xmlstr)
    	tokens = word_tokenize(xmlstr)
    	nostop_tokens = removeStopWords(tokens, type.split("_")[1])    

        these_features.append(extractRepeatation(nostop_tokens, type.split("_")[1]))
		
        cap_word, cap_let = extractCapital(nostop_tokens)
        these_features.append(cap_word)
        these_features.append(cap_let)

        these_features.append(len(tokens))

        LIWCDic = readDictionary(getLIWCOnLang(type.split("_")[1]))
        these_features.extend(extractLIWCFeatures(tokens, LIWCDic))

        these_features.extend(extractReadabilityArray(xmlstr,nostop_tokens))

        these_features.append(extractEmoticons(xmlstr, nostop_tokens))

        these_features.extend(extractHTMLTags(xmlstr, nostop_tokens))

        features.append(these_features)

        #Label each row
        id = file.split(".")[0].split("_")[0]
        if truth_in_xml:
            labels.append(root.attrib["age_group"]+" "+root.attrib["gender"].lower())
            ages.append(root.attrib["age_group"])
            genders.append(root.attrib["gender"].lower())
        else:
            labels.apend(truth_dic[id][1] + " " + truth_dic[id][0].lower())
            ages.append(truth_dic[id][1])
            genders.append(truth_dic[id][0])

        ids.append(id)

    features = numpy.array(features)
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)

    '''
    col_max = [ max(x) for x in zip(*features) ]
    col_min = [ min(x) for x in zip(*features) ]
    for feature in features:
        for obs in range(0,len(feature)):
            val = 0
            try:
                val = (feature[obs] - col_min[obs])/(col_max[obs]-col_min[obs])
            except ZeroDivisionError:
                val = 0
            feature[obs] = float(val)
    '''

    writeCSV(features.tolist(), labels, ages, genders, ids, type+'_features.csv')
		
	#Train liblinear implementation of SVM
    clf = svm.LinearSVC()
    clf.fit(features, labels)  

    joblib.dump(scaler, output_path+type+'_scaler.pkl')
    joblib.dump(topic_model, output_path+type+'_topic_model.pkl')
    joblib.dump(clf, output_path+type+'.pkl') 
    print "All done!"
    print '\a'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains model.')
    parser.add_argument('-i','--input',help='Path to training corpus',required=True)
    parser.add_argument('-o','--output',help='Output path for models',required=True)
    args = parser.parse_args()
    run(args)
    #run("test")