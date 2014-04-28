import argparse, os, csv, string
import xml.etree.ElementTree as ET
from sklearn import svm
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.tag.stanford import POSTagger
from nltk.corpus import cmudict
from LIWC import extractLIWCFeatures, readDictionary, getLIWCOnLang
from Util import removeStopWords, extractLIWCFeatures, extractRepeatation, extractCapital, clean, extractEmoticons, extractHTMLTags
from ReadabilityCalculator import extractReadabilityArray

def run(args):
    #input_path = "/Users/jamarq_laptop/PAN/Software/pan-2/Data/blogs/es/"
    #model_path = "./Models/"
    #output_path = "./Outputs/"

    input_path = args.input
    output_path = args.output
    model_path = args.model

    if not input_path.endswith("/"):
        input_path += "/"
    if not output_path.endswith("/"):
        output_path += "/"
    if not model_path.endswith("/"):
        model_path += "/"

    files = [f for f in os.listdir(input_path) if f.endswith('.xml')]

    tree = ET.parse(input_path + files[0])
    root = tree.getroot()

    type = root.attrib["type"] + "_" + root.attrib["lang"]

    clf = joblib.load(model_path+type+'.pkl')
    topic_model = joblib.load(model_path+type+'_topic_model.pkl')
    scaler = joblib.load(model_path+type+"_scaler.pkl")

    for file in files:

        features = []
        tree = ET.parse(input_path + file)
        root = tree.getroot()

        aut_id = file.split("_")[0]

        if "." in aut_id:
            aut_id = aut_id.split(".")[0]

        lang = type.split("_")[1]
        doccount = tree.find("documents").attrib["count"]

        features.extend(topic_model.get_sim_unseen(input_path+file))

        features.append(int(doccount))

        xmlstr = clean(ET.tostring(root),root.attrib['lang'].lower(),root.attrib['type'],tree)
        xmlstr = filter(lambda x: x in string.printable, xmlstr)
        tokens = word_tokenize(xmlstr)
        nostop_tokens = removeStopWords(tokens, type.split("_")[1])

        features.append(extractRepeatation(nostop_tokens, type.split("_")[1]))

        cap_word, cap_let = extractCapital(nostop_tokens)
        features.append(cap_word)
        features.append(cap_let)

        features.append(len(tokens))

        LIWCDic = readDictionary(getLIWCOnLang(lang))
        features.extend(extractLIWCFeatures(tokens, LIWCDic))

        features.extend(extractReadabilityArray(xmlstr,nostop_tokens))

        these_features.append(extractEmoticons(xmlstr, nostop_tokens))

        these_features.extend(extractHTMLTags(xmlstr, nostop_tokens))

        label = clf.predict(scaler.transform(features))

        author = ET.Element('author')
        author.set("id", aut_id)
        author.set("type", type.split("_")[0])
        author.set("lang", type.split("_")[1])
        author.set("age_group", label[0].split(" ")[0])
        author.set("gender", label[0].split(" ")[1])
        
        ET.ElementTree(author).write(output_path+aut_id+".xml")

    print "All done!"
    print '\a'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predicts labels based on existing model.')
    parser.add_argument('-i','--input',help='Path to test corpus',required=True)
    parser.add_argument('-o','--output',help='Output path for predictions',required=True)
    parser.add_argument('-m','--model',help='Path to model folder',required=True)
    args = parser.parse_args()
    run(args)
    #run("test")