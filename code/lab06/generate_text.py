#!/usr/bin/env python
import numpy as np
import caffe
import json


model_path = "code/lab06/text_gen_deploy.prototxt"
weights_path = "text_gen_iter_4000.caffemodel"

#Net loading parameters changed in Python 3
net = caffe.Net(model_path, 1, weights=weights_path)

caffe.set_mode_gpu()
caffe.set_device(0)

#Gets the json char index map
dict_input_file = "wonderland_dict.json"
char_to_int = json.loads(open(dict_input_file).read())

#And make a reverse char lookup map
int_to_char = dict((i, c) for c, i in char_to_int.items())

num_vocab = len(int_to_char)
print("Total Vocab: ", num_vocab)


test_file = "data/wonderland.txt"

raw_text = open(test_file,"r").read()
seed_text = raw_text[500:4000].lower()


#Predict sequence
seq_length = 50
no_predict = 1000

#Gets the input blobs
input_blob = net.blobs['input_sequence']
cont_blob = net.blobs['cont_sequence']


#Create ndarray for filling
input_np = np.zeros( (seq_length, 1), dtype="float32")
cont_np = np.zeros( (seq_length,1) , dtype="float32")

input_queue= []
input_queue = []
cont_queue = []

#Get seed to fill the string
for i in range(seq_length):
	current_char = seed_text[i]
	input_queue.append(current_char)
	cont_queue.append(1)


result = ""

#Do prediction
for i in range(no_predict):

	#Fill numpy arrays
	for j in range(seq_length):
		input_np[j,0] = char_to_int[input_queue[j]]
		cont_np[j,0] = cont_queue[j]

	#Fill the data blob
	input_blob.data[...] = input_np
	cont_blob.data[...] = cont_np
	output = net.forward()
	output_prob = output['probs']

	#Print prediction sequence
	out_conf = []
	out_seq = []

	conf_hot = np.zeros((num_vocab), dtype="float32")

	for j, p in enumerate(output_prob):
		char_index = p[0].argmax()
		confidence = p[0,char_index]
		out_conf.append(confidence)
		pChar = int_to_char[char_index]
		out_seq.append(pChar)

		#Adds
		conf_hot = np.add(conf_hot, p[0])

	conf_hot = conf_hot / float(len(output_prob))
	out_int = conf_hot.argmax()

	#Last char
	next_char = out_seq[-1]
	next_cont =  1

	input_queue.append(next_char)
	popped_char = input_queue.pop(0)

	cont_queue.append(next_cont)
	cont_queue.pop(0)

	result += popped_char

print(result)




