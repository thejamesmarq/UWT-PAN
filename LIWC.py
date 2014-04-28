from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer

def getLIWCOnLang(lang):
    if lang.lower()=="en":
        return './LIWC/LIWC_English.txt'
    elif lang.lower()=="es":
        return './LIWC/LIWC_Spanish.txt'
    else:
        raise Exception("No dictionary for language " + lang)

#Change based on where you put your dictionary file
def getLIWCEnglishDictionaryPath():
    return './LIWC/LIWC_English.txt'

#Change based on where you put your dictionary file
def getLIWCSpanishDictionaryPath():
    return './LIWC/LIWC_Spanish.txt'

def getEnglishLiwcDic():
    path = getLIWCEnglishDictionaryPath()
    return readDictionary(path)    

def getSpanishLiwcDic():
    path = getLIWCSpanishDictionaryPath()
    return readDictionary(path)

#Create a dictionary from LIWC
def readDictionary(path):
    dic = {}
    with open(path,"r") as text:
        for line in text:
            info = line.strip().split()
            
            if lang == 'en':
                stemmer = SnowballStemmer('english')
            else:
                stemmer = SnowballStemmer('spanish')

            key = stemmer(info[0])
            value = info[1:]
            dic[key]=value
    dic = makeSparseDictionary(dic)
    return dic

#Create a sparse dictionary for LIWC
def makeSparseDictionary(dic):
    for key in dic.keys():
        features = dic[key]
        liwc = initialLIWCList()
        for feature in features:
            value = int(feature)
            liwc[value]=1
        dic[key]=liwc
    return dic
                   
def makeLIWCDictionaryWithHeader(categoryPath, LIWCPath):
    categoryDic = readDictionary(categoryPath)
    LIWCDic = readDictionary(LIWCPath)
    return LIWCDic, categoryDic

def initialLIWCList():
    count = 1
    liwc = []
    while count<70:
        liwc.append(0)
        count = count+1
    return liwc        

#Returns list of LIWC features from list of tokens.
def extractLIWCFeatures(tokens, LIWCDic):
    liwc = initialLIWCList()
    if len(tokens)==0:
        return liwc
    else:
        for token in tokens:
            if unicode(token) in LIWCDic.keys():
                features = LIWCDic[token]
                liwc = [x + y for x, y in zip(liwc, features)]
        #liwc = normalize(liwc, len(tokens))
        return liwc

def normalize(liwc, length):
    return [float(float(item)/float(length)) for item in liwc]
    
