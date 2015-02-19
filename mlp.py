import numpy
import gzip
import pickle
import struct
import pylab
import random
from copy import copy



class Network:
	
	def __init__(self, tailles, saved = False):
		
		self._tailles_couches = tailles
		self._nombre_couches = len(tailles)
		self._liste_matrices = []
		self._liste_biais = []
		
		if saved == True:
			with open("weights", "rb") as file_weights:
				mon_pickler = pickle.Unpickler(file_weights)
				self._liste_matrices = mon_pickler.load()
			
			with open("biases", "rb") as file_biases:
				mon_pickler = pickle.Unpickler(file_biases)
				self._liste_biais = mon_pickler.load()
		
		else:
			i = 1
			while i < self._nombre_couches:
				self._number_current = self._tailles_couches[i]
				self._number_previous = self._tailles_couches[(i-1)]
				matrix = numpy.random.normal(0, 1, (self._number_current, self._number_previous))
				biais = numpy.random.normal(0, 1, self._number_current) 
				self._liste_matrices += [matrix]
				self._liste_biais += [biais]
				i += 1
			
			
	def _get_poids(self):
		return(self._liste_matrices)
		
	def _get_biais(self):
		return(self._liste_biais)
		
		
	def _feedforward(self, input):
		inp = numpy.ndarray.flatten((1/255)*input) 
		N = len(self._liste_matrices)
		i = 0
		
		while i < N:
			inp = numpy.dot(self._liste_matrices[i], inp) + self._liste_biais[i]
			inp = sigmoid_vec(inp)
			i +=1
			
		return inp
		
	

	def _test(self, data):
		if len(data[0]) != len(data[1]):
			raise Exception("Number of images and number of labels don't match !")
		else:
			N = len(data[0])
			number_success = 0
			i = 0
			while i < N:
				result = self._feedforward(data[0][i])
				number = result.argmax()
				if number == data[1][i]:
					number_success += 1
					
				if i% 100 == 0:
					print("Example {}".format(i))
					
				i += 1
					
			print("Accuracy : {}/{}".format(number_success, N))
			
			return number_success/N
	
	def _backpropagation(self, input, true_output):
		inp = input
		N = len(self._liste_matrices)
		i = 0
		list_wsum = []
		list_outputs = [input]
		
		while i < N:
			inp = numpy.dot(self._liste_matrices[i], inp) + self._liste_biais[i]
			list_wsum += [inp]
			inp = sigmoid_vec(inp)
			list_outputs += [inp]
			i += 1
			
		list_vec_errors = [(inp - true_output)*sigmoid_prime_vec(list_wsum[(N-1)])]
		i = 1
		while i < N:
			transp = numpy.transpose(self._liste_matrices[(N-i)])
			error = numpy.dot(transp, list_vec_errors[0])*sigmoid_prime_vec(list_wsum[(N-i-1)])
			list_vec_errors = [error] + list_vec_errors
			i += 1
		

		list_part_deriv = []
		list_part_bias = []
		l = 0
		while l < N:
			matrice = copy(self._liste_matrices[l])
			bias = copy(self._liste_biais[l])
			for i in range(0, len(matrice[:, 1])):
				for j in range(0, len(matrice[1, :])):
					matrice[i, j] = list_outputs[l][j]*list_vec_errors[l][i]
				
				bias[i] = list_vec_errors[l][i]
				
			list_part_deriv += [matrice]
			list_part_bias += [bias]
			l += 1
			
		
		return (list_part_deriv, list_part_bias)
		
		
	def _update_mini_batch(self, list_images, list_labels, rate):
		N = len(list_images)
		Nb_layers = self._nombre_couches
		if N != len(list_labels):
			raise Exception("Number of images and labels doesn't match")
		else:
			list_mat = self._liste_matrices.copy()
			list_bias = self._liste_biais.copy()
				
			i = 0
			while i < N:
				desired_output = numpy.zeros(self._tailles_couches[-1])
				desired_output[list_labels[i]] = 1
				matrices, bias = self._backpropagation(numpy.ndarray.flatten((1/255)*list_images[i]), desired_output)
				for l, elt in  enumerate(matrices):
					list_mat[l] += (-rate/N)*elt
					list_bias[l] += (-rate/N)*bias[l]
					
				i += 1
			
			self._liste_matrices = list_mat.copy()
			self._liste_biais = list_bias.copy()
	
		
		
	def stochastic_gradient(self, training_set, number_epochs, size_mini_batch, rate):
		if len(training_set[0])% size_mini_batch != 0:
			raise Exception("Size of training set is not a multiple of mini-batch size !")
		else:
			training_images = training_set[0]
			training_labels = training_set[1]
		
			i = 0
			while i < number_epochs:
				temp_images = copy(training_images)
				temp_labels = copy(training_labels)
				j = 0
				m = len(training_set[0])/size_mini_batch
				print(m)
				while j < m:
					mini_batch_images = []
					mini_batch_labels = []
					l = 0
					while l < size_mini_batch:
						random_index = random.randint(0, (len(temp_images)-1))
						mini_batch_images += [temp_images[random_index]]
						mini_batch_labels += [temp_labels[random_index]]
						del temp_images[random_index]
						del temp_labels[random_index]
						l += 1
					self._update_mini_batch(mini_batch_images, mini_batch_labels, rate)
					j +=1
					
				print("Epoch {} complete".format(i+1))
				i += 1
				
			with open("weights", "wb") as fichier_weights:
				my_pickler = pickle.Pickler(fichier_weights)
				my_pickler.dump(self._liste_matrices)
				
			with open("biases", "wb") as fichier_biases:
				my_pickler = pickle.Pickler(fichier_biases)
				my_pickler.dump(self._liste_biais)
				
	liste_matrices = property(_get_poids)
	
	
	
	
	
	
	
	
	
	
def sigmoid(z):
	return(1.0/(1.0+numpy.exp(-z)))
	
sigmoid_vec = numpy.vectorize(sigmoid)
			

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))
	
sigmoid_prime_vec = numpy.vectorize(sigmoid_prime)



def load_training_data():
	train = gzip.open("train-images-idx3-ubyte.gz", "rb")
	labels = gzip.open("train-labels-idx1-ubyte.gz", "rb")
	
	train.read(4)
	labels.read(4)
		
	number_images = train.read(4)
	number_images = struct.unpack(">I", number_images)[0]
		
	rows = train.read(4)
	rows = struct.unpack(">I", rows)[0]
		
	cols = train.read(4)
	cols = struct.unpack(">I", cols)[0]
		
	number_labels = labels.read(4)
	number_labels = struct.unpack(">I", number_labels)[0]
	
	image_list = []
	label_list = []
	if number_images != number_labels:
		raise Exception("The number of labels doesn't match with the number of images")
	else:
		for l in range(number_labels):
			if l % 1000 == 0:
				print("l:{}".format(l))
				
			mat = numpy.zeros((rows, cols), dtype = numpy.uint8)
			for i in range(rows):
				for j in range(cols):
					pixel = train.read(1)
					pixel = struct.unpack(">B", pixel)[0]
					mat[i][j] = pixel
					
			
			image_list += [mat]
			lab = labels.read(1)
			lab = struct.unpack(">B", lab)[0]
			label_list += [lab]
		
	
	train.close()
	labels.close()
		
	return ([image_list, label_list])
	










def load_test_data():
	test = gzip.open("t10k-images-idx3-ubyte.gz", "rb")
	labels = gzip.open("t10k-labels-idx1-ubyte.gz", "rb")
	
	test.read(4)
	labels.read(4)
		
	number_images = test.read(4)
	number_images = struct.unpack(">I", number_images)[0]
		
	rows = test.read(4)
	rows = struct.unpack(">I", rows)[0]
		
	cols = test.read(4)
	cols = struct.unpack(">I", cols)[0]
		
	number_labels = labels.read(4)
	number_labels = struct.unpack(">I", number_labels)[0]
	
	image_list = []
	label_list = []
	if number_images != number_labels:
		raise Exception("The number of labels doesn't match with the number of images")
	else:
		for l in range(number_labels):
			if l % 1000 == 0:
				print("l:{}".format(l))
				
			mat = numpy.zeros((rows, cols), dtype = numpy.uint8)
			for i in range(rows):
				for j in range(cols):
					pixel = test.read(1)
					pixel = struct.unpack(">B", pixel)[0]
					mat[i][j] = pixel
					
			
			image_list += [mat]
			lab = labels.read(1)
			lab = struct.unpack(">B", lab)[0]
			label_list += [lab]
		
	
	test.close()
	labels.close()
		
	return ([image_list, label_list])	
	



	
def view(image, label=""):
	print("Number : {}".format(label))
	pylab.imshow(image, cmap = pylab.cm.gray)
	pylab.show()
	
a = load_test_data()



net = Network([28*28, 30, 10], True)

result = net._test(a)
	
	
