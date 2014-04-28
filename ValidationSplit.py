from sklearn.cross_validation import ShuffleSplit
import argparse, os, itertools, shutil

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Splits corpus into testtrain and validation sets')
	parser.add_argument('-i','--input',help='Path to corpus',required=True)
	args = parser.parse_args()
	input_path = args.input

	if not input_path.endswith("/"):
		input_path += "/"

	if not os.path.exists(input_path+"testtrain"):
		os.makedirs(input_path+"testtrain")
	if not os.path.exists(input_path+"validation"):
		os.makedirs(input_path+"validation")

	shutil.copy(input_path+"truth.txt",input_path+"testtrain")
	shutil.copy(input_path+"truth.txt",input_path+"validation")

	files = [f for f in os.listdir(input_path) if f.endswith('.xml')]

	truth_dic = dict()
	split_labels = dict()
	labels = []

	with open(input_path+"truth.txt","rb") as truth_in:
		for line in truth_in:
			truth_data = line.split(":::")
			label = truth_data[2]+" "+truth_data[1]
			if label not in truth_dic:
				truth_dic[label] = []
			truth_dic[label].append(truth_data[0])

	print len(truth_dic)

	for key, value in truth_dic.iteritems():
		ss = ShuffleSplit(len(value), n_iter=1, test_size=0.1,indices=False)
		print "splitting "+key+" files"
		testtrain_files = None
		validation_files = None

		for testtrain, validation in ss:
			#print len(testtrain), len(validation)
			testtrain_files = list(itertools.compress(truth_dic[key],testtrain))
			validation_files = list(itertools.compress(truth_dic[key],validation))

		print "moving testtrain files for " + key
		for file in testtrain_files:
			shutil.move(input_path+file+".xml",input_path+"testtrain/"+file+".xml")
		print "moving validation files for " + key
		for file in validation_files:
			shutil.move(input_path+file+".xml",input_path+"validation/"+file+".xml")
			





