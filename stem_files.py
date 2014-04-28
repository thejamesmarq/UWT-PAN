import os, string, argparse
from nltk.stem.snowball import SnowballStemmer

def stemFiles(args):
	folder = args.input
	stem_path = args.output
	lang = args.lang

	#create output directory
	if not os.path.exists(stem_path):
		os.makedirs(stem_path)

	#set language of stemmer
	if lang == 'en':
		stemmer = SnowballStemmer('english')
	elif lang == 'es':
		stemmer = SnowballStemmer('spanish')
	else:
		raise Exception("Must use english or spanish")

	#get list of files
	files = [f for f in os.listdir(folder) if f.endswith('.txt')]

	for file in files:
		with open(folder+file, "r") as infile, open(stem_path+file, 'wb') as outfile:
			data=infile.read().replace('\n', ' ').replace('\t',' ')
			data=' '.join(data.split())

			data = filter(lambda x: x in string.printable, data)
			
			exclude = set(string.punctuation)
			data = ''.join(ch for ch in data if ch not in exclude)

			out_string = ''

			for word in data.split(" "):
				if len(word)==0:
					continue
				out_string += " " + stemmer.stem(word)

			outfile.write(out_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stems all files in a directory.')
    parser.add_argument('-i','--input',help='Path to unstemmed files',required=True)
    parser.add_argument('-o','--output',help='Output path for stemmed files',required=True)
    parser.add_argument('-l','--lang',help='Language of documents (en|es)',required=True)
    args = parser.parse_args()
    stemFiles(args)