import numpy as np
import matplotlib.pyplot as plt

def plot_misclassifications(x, y, y_prob, file_prefix="i"):
	#x = array of images tested.  Outermost index is image number
	#y = true labels
	#y_prob = estimated probability that each image is of class 1

	#if len(x.shape)>3:
	#	x = np.squeeze(x[:,:,:,0])

	print(x.shape)
	y_hat = np.round(y_prob) #make a hard decision
	false_positives_idx = np.nonzero(np.logical_and(y_hat == 1, y == 0))[0]
	false_negatives_idx = np.nonzero(np.logical_and(y_hat == 0, y == 1))[0]
	print(false_positives_idx)
	print('Holdout set had {} false positives and {} false negatives'.format(false_positives_idx.size, false_negatives_idx.size))

	# Plot the false positives
	plt.figure(figsize=(8,8))
	print('False Positives')
	false_positives_to_plot = min(16, false_positives_idx.size)
	grid_dim = np.ceil(np.sqrt(false_positives_to_plot))
	for i in range(0, false_positives_to_plot):
	    plt.subplot(grid_dim,grid_dim,i+1)
	    plt.imshow(np.squeeze(x[false_positives_idx[i], :, :, :]), cmap='gray')
	    plt.title(np.round(y_prob[false_positives_idx[i]], decimals=2))
	    plt.axis('off')
	plt.savefig('{}_false_positive.png'.format(file_prefix))
	plt.show()

	# Plot the false negatives
	plt.figure(figsize=(8,8))
	print('False Negatives')
	false_negatives_to_plot = min(16, false_negatives_idx.size)
	grid_dim = np.ceil(np.sqrt(false_negatives_to_plot))
	for i in range(0, false_negatives_to_plot):
	    plt.subplot(grid_dim,grid_dim,i+1)
	    plt.imshow(np.squeeze(x[false_negatives_idx[i], :, :, :]), cmap='gray')
	    plt.title(np.round(y_prob[false_negatives_idx[i]], decimals=2))
	    plt.axis('off')
	plt.savefig('{}_false_negative.png'.format(file_prefix))
	plt.show()