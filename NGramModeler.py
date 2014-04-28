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
from sklearn import preprocessing

def clean(text,lang):
    start_index = '<'
    finish_index = '>'
    text = text.replace('&lt;','<')
    text = text.replace('&gt;','>')
    starter = text.find(start_index)
    while starter<>-1:
        data_index = text.find('![CDATA[')
        if data_index<>-1:
            text = text.replace('<![CDATA[', '')
            text = text.replace(']]>', '')
            starter = text.find(start_index)
        else:
            finisher = text.find(finish_index,starter+1)
            if  finisher<>-1:
                rm_text = text[starter:finisher+1]
                text = text.replace(rm_text,' ')
                starter = text.find(start_index)
            else:
                starter =-1
    text = text.replace(']]>','')
    text = text.replace('\n',' ')
    text = text.strip() 
    
    text=cleanSpams(text) 
    text = cleanHtmlTags(text)
    text = stem(text,lang)
    return text
        
def stem(text,lang):

    text = filter(lambda x: x in string.printable, text)

    if lang == 'en':
        stemmer = SnowballStemmer('english')
    elif lang == 'es':
        stemmer = SnowballStemmer('spanish')
    else:
        raise Exception("Must use english or spanish")

    out_string = ''

    for word in text.split(" "):
        if len(word)==0:
            continue
        try:
            out_string += " " + stemmer.stem(word)
        except IndexError:
            continue

    return out_string

def cleanHtmlTags(text):
    #print 'html'
    start = text.find('&')
    finish = start+20
    while  start <>-1 and (finish-start)<10:
        finish = text.find(';', start+1)
        if finish<>-1:
            rm = text[start:finish+1]
            text = text.replace(rm,' ')
        start = text.find('&')   
    return text
   
def cleanSpams(text):
    #print 'spam'
    start = text.find('%')
    while start <>-1:
        finish= text.find('%', start+1)
        index = start
        while finish<>-1 and float(finish-index) <5.0:
            index = finish
            finish = text.find('%', index+1)
        rm = text[start:index+1]
        text = text.replace(rm,' ')
        start = text.find('%')
    return text

def usingVectorizers(input_path, label_type, k_feat, the_texts):
    print "starting for " +label_type+ " with "+str(k_feat)+" selected features"
    files = [f for f in os.listdir(input_path) if f.endswith('.xml')]
    
    #PAN 14
    type = input_path.split("/")[-3]+"_"+input_path.split("/")[-2]

    #PAN 13
    #type = 'pan13' + files[0].split("_")[-3]

    texts = the_texts
    labels = []
    print "texts are size " +str(len(texts))
    for file in files:
        fileinfo = file.split("_")
        
        '''
        Label combined, gender, or age
        '''
        if label_type == "comb":
            labels.append(fileinfo[2] +" "+ fileinfo[3].split(".")[0])
        elif label_type == "age":
            labels.append(fileinfo[3].split(".")[0])
        elif label_type == "gender":
            labels.append(fileinfo[2])

    del files
    gc.collect()

    print "finished preprocessing"
    print '\a'

    #ngram_vectorizer = HashingVectorizer(non_negative = True, ngram_range=(1, 2),token_pattern=r'\b\w+\b',binary=True)
    ngram_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2),token_pattern=r'\b\w+\b',binary=True)
    #ngram_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),token_pattern=r'\b\w+\b')

    print "vectorizing"
    grams = ngram_vectorizer.fit_transform(texts)
    grams = grams.toarray()

    with open("./Vocabs/"+type.split("_")[0]+"/"+type.split("_")[1]+"/"+type+"_possessives_out.txt", "wb") as possout:
        poss_writer = csv.writer(possout)

        for gram in ngram_vectorizer.get_feature_names():
            gram_first = gram.split(" ")[0]
            if gram_first =="my" or gram_first == "mi":
                possout.write(gram+"\n")

    del texts
    gc.collect()

    print grams.nbytes
    print grams.shape

    print "selecting features"
    ch2 = SelectKBest(chi2, k=k_feat)
    #ch2 = SelectKBest(f_classif, k=50)
    grams = ch2.fit_transform(grams, labels)
    header = range(1,grams.shape[1]+1)
    header.append("labels")

    print grams.nbytes
    print grams.shape

    with open("./Vocabs/"+type.split("_")[0]+"/"+type.split("_")[1]+"/"+type+"_"+label_type+"_"+str(k_feat)+"_ngrams.csv", "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(list(itertools.compress(ngram_vectorizer.get_feature_names(),ch2.get_support())))

        '''
        writer.writerow(header)
        index = 0
        for row in grams:
            written = row.tolist()
            written.append(labels[index])
            writer.writerow(written)
            index+=1
        '''

    print "All done!"
    print '\a'

'''
def usingBigramExtractor(input_path):
    files = [filter(lambda x: x in string.printable, clean(open(input_path+f,"rb").read())) for f in os.listdir(input_path) if f.endswith('.xml')]
    print "read done!"
    print '\a'

    b_grams = set()
    for file in files:
        b_grams.update(bigrams(word_tokenize(file)))

    del files

    features = []

    files = [f for f in os.listdir(input_path) if f.endswith('.xml')]

    for file in files:
        with open(input_path+file) as checkfile:
            these_features = []
            check_str = bigrams(word_tokenize(filter(lambda x: x in string.printable, clean(checkfile.read()))))
            for check_bgram in b_grams:
                these_features.append(check_str.count(check_bgram))
            features.append(these_features)
                
    print "All done!"
    print '\a'
'''

if __name__ == "__main__":
    
    #PAN 14
    types = ['socmed/en/']
    
    #PAN 13
    #types = ['pan13/en/','pan13/es/']

    for type in types: 
        #PAN 14
        input_path = "/Users/itadmin/PAN/Data/pan14/pan14-author-profiling-training-corpus-2014-04-09/v1/"+type
        
        #PAN 13
        #input_path = "/Users/itadmin/PAN/Data/"+type

        for k_feat in range(1000,5001,1000):
            #usingBigramExtractor(input_path)

            texts = []
            files = [f for f in os.listdir(input_path) if f.endswith('.xml')]

            for file in files:

                with open(input_path+file, "rb") as curfile:
                    content = curfile.read()
                    texts.append(filter(lambda x: x in string.printable, clean(content,input_path.split("/")[-2])))

            usingVectorizers(input_path, "comb", k_feat,texts)
            usingVectorizers(input_path, "age", k_feat,texts)
            usingVectorizers(input_path, "gender", k_feat,texts)


    