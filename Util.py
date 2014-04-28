import argparse, os, csv, string, re
import xml.etree.ElementTree as ET
from sklearn import svm
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.tag.stanford import POSTagger
from nltk.corpus import cmudict
from LIWC import extractLIWCFeatures, readDictionary, getLIWCOnLang
import xml.etree.ElementTree as ET
import os, argparse, string
from nltk.stem.snowball import SnowballStemmer

def selectIndexes(texts,indexes):
    results = []
    for s in indexes:
        results.append(texts[s])
    return results

def extractEmoticons(text, tokens):
    matches = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    if len(tokens)>0:
        return float(float(len(matches))/float(len(tokens)))
    else:
        return 0

def extractHTMLTags(text, tokens):
    docArray = []
    if len(tokens)>0:
        docArray.append(float(float(len(re.findall(r'<a',text)))/float(len(tokens))))
        docArray.append(float(float(len(re.findall(r'<img',text)))/float(len(tokens))))
        docArray.append(float(float(len(re.findall(r'<b',text)))/float(len(tokens))))
        docArray.append(float(float(len(re.findall(r'<i',text)))/float(len(tokens))))
        docArray.append(float(float(len(re.findall(r'<ui',text)) + len(re.findall(r'<ol',text)))/float(len(tokens))))
    else:
        docArray.append(0)
        docArray.append(0)
        docArray.append(0)
        docArray.append(0)
        docArray.append(0)
    return docArray   

def removeStopWords(tokens, lang):
    filteredToken=tokens
    if lang =='en':
        filteredToken = [w for w in tokens if not w in stopwords.words('english')]
    elif lang =='es':
        filteredToken = [w for w in tokens if not w in stopwords.words('spanish')]
    return filteredToken

def extractEmoticons(text, tokens):
    matches = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    if len(tokens)>0:
        return float(float(len(matches))/float(len(tokens)))
    else:
        return 0

def extractRepeatation(tokens, text):
    number = 0
    for word in tokens:
        number = text.count(word)
    if len(tokens)>0:
        frequencyOfRepeatation = float(float(number)/float(len(tokens)))
    else:
        frequencyOfRepeatation=0
    return frequencyOfRepeatation

def extractCapital(tokens):
    capitalLetter = 0
    capitalWord = 0
    wholeSize = 0
    for token in tokens:
        size = len(token)
        wholeSize= wholeSize+size
        uppers = [l for l in token if l.isupper()]
        length = len(uppers)
        capitalLetter = capitalLetter+length
        if size == length:
            capitalWord = capitalWord+1
    if len(tokens)>0:
        return float(float(capitalWord)/float(len(tokens))), float(float(capitalLetter)/float(wholeSize))
    else:
        return 0,0

def clean(text,root):
    lang = root.attrib['lang']

    if root.attrib['type'] == 'twitter':
        for child in root[0]:
            if "RT " in child.text:
                root[0].remove(child)
        text = ET.tostring(root)

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
    else:
        stemmer = SnowballStemmer('spanish')

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

def writeCSV(features,labels,ages,genders,ids,path):
    header = range(0,len(features[0]))
    header.append("labels")
    header.append("ages")
    header.append("genders")
    header.append("doc-id")
    with open(path,"wb") as outpath:
        writer = csv.writer(outpath)
        writer.writerow(header)
        for feat, lab, age, gend, id in zip(features, labels, ages, genders, ids):
            row = []
            row.extend(feat)
            row.append(lab)
            row.append(age)
            row.append(gend)
            row.append(id)
            writer.writerow(row)

