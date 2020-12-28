import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:200]
y = y_train[:200]

x = x_train.T
x = x/255

y.resize((200,1))
y = y.T

print(pd.Series(y[0]).value_counts())

# converting into binary classification
for i in range(y.shape[1]):
	if y[0][i] > 4:
		y[0][i] = 1
	else:
		y[0][i] = 0

print(pd.Series(y[0]).value_counts())

#initializing filter
f = np.random.uniform(size = (3,5,5))
f = f.T

print('Filter 1', '\n', f[:,:,0], '\n')
print('Filter 2', '\n', f[:,:,1], '\n')
print('Filter 3', '\n', f[:,:,2], '\n')

print(x.shape, y.shape, f.shape)

#Generating patches from images
new_image = []

#for no.of images
for k in range(x.shape[2]):
	#sliding in horizontal direction
	for i in range(x.shape[0]-f.shape[0]+1):
		#sliding in vertical direction
		for j in range(x.shape[1]-f.shape[1]+1):
			new_image.append(x[:,:,k][i:i+f.shape[0],j:j+f.shape[1]])

#resizing the generated patches as per no.of images
new_image = np.array(new_image)
new_image.resize((x.shape[2], int(new_image.shape[0]/x.shape[2]), new_image.shape[1], new_image.shape[2]))
print(new_image.shape)


#no.of features in dataset
s_row = x.shape[0] - f.shape[0] + 1
s_col = x.shape[1] - f.shape[1] + 1
num_filter = f.shape[2]

input_layer_neurons = (s_row) * (s_col) * num_filter
output_neurons = 1

#initializing weight
wo = np.random.uniform(size = (input_layer_neurons, output_neurons))
print(wo)

#sigmoid function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

#derivative of sigmoid function
def derivatives_sigmoid(x):
	return x * (1 - x)

#generating o/p of convolution layer
filter_output = []
#for each image
for i in range(len(new_image)):
	#apply each filter
	for k in range(f.shape[2]):
		#do element wise multiplication
		for j in range(new_image.shape[1]):
			filter_output.append((new_image[i][j] * f[:,:,k]).sum())

filter_output = np.resize(np.array(filter_output), (len(new_image), f.shape[2], new_image.shape[1]))

#applying activation over convolution o/p
filter_output_sigmoid = sigmoid(filter_output)
print(filter_output.shape, filter_output_sigmoid.shape)

#generating input for fully connected layer
filter_output_sigmoid = filter_output_sigmoid.reshape((filter_output_sigmoid.shape[0], filter_output_sigmoid.shape[1]*filter_output_sigmoid.shape[2]))
filter_output_sigmoid = filter_output_sigmoid.T

#Linear transformation for fully connected layer
output_layer_input = np.dot(wo.T, filter_output_sigmoid)
output_layer_input = (output_layer_input - np.average(output_layer_input))/ np.std(output_layer_input)

#activation function
output = sigmoid(output_layer_input)

#Error 
error = np.square(y-output)/ 2

#Error wrt output gradient
error_wrt_output = - (y-output)

#Error wrt sigmoid transformation (output_layer_input)
output_wrt_output_layer_input = output * (1 - output)

#Error wrt weight
output_wrt_w = filter_output_sigmoid

#Error wrt sigmoid output
output_layer_input_wrt_filter_output_sigmoid = wo.T

#Error wrt sigmoid transformation
filter_output_sigmoid_wrt_filter_output = filter_output_sigmoid + (1 - filter_output_sigmoid)

#Calculating derivatives for backprop convolutions
error_wrt_filter_output = np.dot(output_layer_input_wrt_filter_output_sigmoid.T, error_wrt_output * output_wrt_output_layer_input) * filter_output_sigmoid_wrt_filter_output
error_wrt_filter_output = np.average(error_wrt_filter_output, axis=1)

error_wrt_filter_output = np.resize(error_wrt_filter_output, (x.shape[0]-f.shape[0]+1, x.shape[1]-f.shape[1]+1, f.shape[2]))


filter_update = []

for i in range(f.shape[2]):
	for j in range(f.shape[0]):
		for k in range(f.shape[1]):
			temp = 0
			spos_row = j
			spos_col = k
			epos_row = spos_row + s_row
			epos_col = spos_col + s_col
			for l in range(x.shape[2]):
				temp = temp + (x[spos_row:epos_row, spos_col:epos_col, l] * error_wrt_filter_output[:,:,i]).sum()
				filter_update.append(temp/ x.shape[2])

filter_update_array = np.array(filter_update)
filter_update_array = np.resize(filter_update_array, (f.shape[2], f.shape[0], f.shape[1]))

lr = 0.1
for i in range(f.shape[2]):
	f[:,:,1] = f[:,:,1] - lr*filter_update_array[i]

print(f.shape)











