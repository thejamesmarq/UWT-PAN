import random, os
import xml.etree.ElementTree as ET

if __name__ == "__main__":

	#the actual age and gender of all documents
	truth_dic = dict()

	#path where your corpora are
	input_path = '/Users/jamarq_laptop/PAN/Software/pan-2/Data/'

    #directories containing your corpora
	corps = ['socmed/en/','blogs/en/','reviews/en/','twitter/en/','socmed/es/','blogs/es/','twitter/es/']
	
	#labels detect (specific to genders)
	genders = ['male','female']

	#vocabulary to label as male or female
	rules_male = ['my wife','fuck','fucking','xbox','himself','shit','ps3']
	rules_female = ['love','cute','<3',':-)']

	correct = 0
	total = 0
	hit_correct = 0
	hit_wrong = 0

	path = input_path+corps[0]

	with open(path+"truth.txt") as truthfile:
		for line in truthfile:
			data = line.split(":::")
			truth_dic[data[0]]=(data[1],data[2])

	files = [f for f in os.listdir(path) if f.endswith('.xml')]

	predicted_gender = [None] * len(files)
	actual_gender = [None] * len(files)
    
	index = 0
	for file in files:
		truth = truth_dic[file.split(".")[0]]
		tree = None
		try:
			tree = ET.parse(path+file)
		except ET.ParseError:
			print "bad xml"
			continue
		root = tree.getroot()
		lang = root.get('lang').lower()
		genre = root.get('type').lower()
		age = root.get('age_group')
		gender = root.get('gender').lower()

		actual_gender[index] = gender

   		infile = open(path+file,'rb')
   		text = infile.read()	
   		for rule in rules_male:
   			if rule in text:
   				if gender == 'male':
   					hit_correct+=1
   				else:
   					hit_wrong+=1
   				predicted_gender[index] = 'male'
   				break
   		if predicted_gender == None:
   			for rule in rules_female:
   				if rule in text:
   					if gender == 'female':
   						hit_correct+=1
   					else:
   						hit_wrong+=1
   					predicted_gender[index] = 'female'
   					break
   		infile.close()
   		if predicted_gender[index] == None:
			predicted_gender[index] = random.choice(genders)
		index+=1

	for pred, act in zip(predicted_gender, actual_gender):
   		if pred == act:
   			correct += 1
   		total += 1

   	print hit_correct
   	print hit_wrong
   	print correct
   	print total
   	print "accuracy: " + str(correct/float(total))



