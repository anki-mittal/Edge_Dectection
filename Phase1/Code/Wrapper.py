#!/usr/bin/env python3

"""
RBE/CS549 Spring 2024: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

from cgitb import text
import numpy as np
import cv2
from matplotlib import image
import skimage.transform as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def Image_convolve(image, kernel, padding=0, stride =1):
	kernel = np.flipud(np.fliplr(kernel))
	kernel_x, kernel_y = kernel.shape
	image_height = image.shape[0]
	image_width = image.shape[1]
	padding = int((kernel_x - 1)/2)
	output_height = int((image_height + 2*padding - kernel_x)/stride + 1)
	output_width = int((image_width + 2*padding - kernel_y)/stride + 1)

	convolved_image = np.zeros((output_height,output_width))

	padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding,cv2.BORDER_DEFAULT)
	
	for y in range(image.shape[1]):
		if y>image.shape[-1] - kernel_y:
			break
		if y%stride == 0:
			for x in range (image.shape[0]):
				if x>image.shape[0] - kernel_x:
					break
				try:
					if x%stride == 0:
						convolved_image[x,y] = (kernel * padded_image[x:x+kernel_x, y:y+kernel_y]).sum()
				except:
					break
	return convolved_image

def images_plot(fig_size, filters, x_len, y_len, name):
	fig = plt.figure(figsize = fig_size)
	total_filters = len(filters)
	for i in np.arange(total_filters):
		ax = fig.add_subplot(y_len, x_len, i+1, xticks = [], yticks = [])
		plt.imshow(filters[i], cmap = 'gray')
	plt.savefig(name)
	plt.close()

def Gauss_derivative(sigma, x, d_order):
	gauss = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
	if d_order == 1:        
		return -gauss*(x/sigma**2)
	elif d_order == 2:        
		return gauss*((x**2-sigma**2)/sigma**4)
	else:
		return gauss

# 2 Dim gaussian for different x and y sigma
def	Gauss_oval(sigma, kernelsize, d_order_x, d_order_y):
	interval = kernelsize/2.5
	[x, y] = np.meshgrid(np.linspace(-interval, interval, kernelsize),
						np.linspace(-interval,interval,kernelsize))
	grid = np.array([x.flatten(), y.flatten()])
	gauss_x = Gauss_derivative(3*sigma, grid[0,...], d_order_x)
	gauss_y = Gauss_derivative(sigma, grid[1,...], d_order_y)
	gauss = gauss_x * gauss_y
	filter = np.reshape(gauss, (kernelsize, kernelsize))
	return filter

# 2 Dim gaussian for same x and y sigma
def get_gaussian(sigma, kernelsize):
	interval = kernelsize/2
	x, y = np.meshgrid(np.linspace(-interval, interval, kernelsize),
						np.linspace(-interval,interval,kernelsize))
	gaussian = 1 / (2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
	return gaussian


def Laplacian(sigma, kernelsize):
	laplacian_filter = np.array([[0, 1, 0],
								[1, -4, 1],
								[0, 1, 0]])
	gauss = get_gaussian(sigma, kernelsize)
	log = Image_convolve(gauss, laplacian_filter)
	return log


def DOG_filters(sigma, kernel_size, no_of_orientations, sobel_vertical):
		dog_filters = []
		orientations = np.linspace(0,360,no_of_orientations)                
		for x in sigma:
			gauss_kernel = get_gaussian(x, kernel_size)
			sobel_convolve = Image_convolve(gauss_kernel, sobel_vertical)
			for i in range(0, no_of_orientations):
				filter = tf.rotate(sobel_convolve, orientations[i])
				dog_filters.append(filter)
		return dog_filters


def LM_filters(sigma, kernelsize):
	filters = []
	orientations = np.linspace(0,180,6)
	i = len(sigma)-2
	while i>= 0:
		gauss_kernel = Gauss_oval(sigma[i], kernelsize, 0, 1)     
		for j in range(0,len(orientations)):
			filter = tf.rotate(gauss_kernel, orientations[j])
			filters.append(filter)
		gauss_kernel = Gauss_oval(sigma[i], kernelsize, 0, 2)    
		for j in range(0,len(orientations)):
			filter = tf.rotate(gauss_kernel, orientations[j])
			filters.append(filter)
		i = i -1
	for i in range(0,len(sigma)):
		filters.append(Laplacian(sigma[i], kernelsize))             
	for i in range(0,len(sigma)):
		filters.append(Laplacian(3*sigma[i], kernelsize))
	for i in range(0,len(sigma)):
		filters.append(get_gaussian(sigma[i], kernelsize))       
	return filters


# code reference is taken from wikipedia
def Gabor(sigma, theta, Lambda, psi, gamma):
	sigma_x = sigma
	sigma_y = float(sigma)/gamma
	nstds = 3 
	xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
	xmax = np.ceil(max(1, xmax))
	ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
	ymax = np.ceil(max(1, ymax))
	xmin = -xmax
	ymin = -ymax
	(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
	x_theta = x * np.cos(theta) + y * np.sin(theta)
	y_theta = -x * np.sin(theta) + y * np.cos(theta)
	gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
	return gb


def gabor_filters(sigma, kernelsize, theta, Lambda, psi, gamma, number):
	filters = []
	orientations = np.linspace(90,270,number)
	for i in range(0,len(sigma)):
		gabor_kernel = Gabor(sigma[i], theta, Lambda, psi, gamma)
		for j in range(0, number):
			filter = tf.rotate(gabor_kernel, orientations[j])
			filters.append(filter)
	return filters


def convolve_filters(image, filter_bank):
	t_map = np.array(image)
	for i in range(0, len(filter_bank)):
		filter = np.array(filter_bank[i])
		filter_map = Image_convolve(image, filter)
		t_map = np.dstack((t_map, filter_map))
	return t_map

#create texton maps
def Texton_map(img, dogfilters, lmfilters, gaborfilters, clusters):
	textonmap_dog = convolve_filters(img, dogfilters)       
	textonmap_lm = convolve_filters(img, lmfilters)         
	textonmap_gabor = convolve_filters(img, gaborfilters)   
	textonmap = np.dstack((textonmap_dog, textonmap_lm, textonmap_gabor))         
	tex = np.reshape(textonmap, ((img.shape[0]*img.shape[1]),textonmap.shape[2]))   
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(tex)      
	pred = kmeans.predict(tex)
	prediction = np.reshape(pred, (img.shape[0],img.shape[1]))
	return prediction

#to create brightness maps
def Brightness_map(img, clusters):       
	img = np.array(img)
	image = np.reshape(img, ((img.shape[0]*img.shape[1]),1))
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image)
	pred = kmeans.predict(image)                                 
	prediction = np.reshape(pred, (img.shape[0],img.shape[1]))
	return prediction

def color_map(img, clusters):       
	img = np.array(img)
	image = np.reshape(img, ((img.shape[0]*img.shape[1]),img.shape[2]))
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image)
	pred = kmeans.predict(image)                                 
	prediction = np.reshape(pred, (img.shape[0],img.shape[1]))
	return prediction

def Halfdisc_Masks(scales):
	halfdiscs = []
	angles = [0, 180, 30, 210, 45, 225, 60, 240, 90, 270, 120, 300, 135, 315, 150, 330]          
	no_of_disc = len(angles)
	for r in scales:
		kernelsize = 2*r + 1
		cc = r
		kernel = np.zeros([kernelsize, kernelsize])
		for i in range(r):
			for j in range(kernelsize):
				a = (i-cc)**2 + (j-cc)**2                                    
				if a <= r**2:
					kernel[i,j] = 1
		
		for i in range(0, no_of_disc):                                       
			mask = tf.rotate(kernel, angles[i])
			mask[mask<=0.5] = 0
			mask[mask>0.5] = 1
			halfdiscs.append(mask)
	return halfdiscs

def chi_square_distance(map, bins, mask, inv_mask):
	chi_sqr_dist = map*0
	for i in range(0, bins):
		tmp = np.zeros_like(map)
		tmp[map == i] = 1
		g_i = cv2.filter2D(tmp, -1, mask)
		h_i = cv2.filter2D(tmp, -1, inv_mask)
		chi_sqr_dist = chi_sqr_dist + ((g_i - h_i)**2)/(g_i + h_i + 0.01)   #chi-square distance formula (0.001 added so that value does not become Nan in case denominator becomes 0)
	return chi_sqr_dist/2

def Gradients(map, bins, filters):
	a, b = map.shape
	grad = np.array(map)
	i = 0
	while i < len(filters)-1:
		chi_sqr_dist = chi_square_distance(map, bins, filters[i], filters[i+1])      #chi-square distance calculate using two opposite half disc masks
		grad = np.dstack((grad, chi_sqr_dist))                                       #stack all chi-square distances 
		i += 2
	gradient = np.mean(grad, axis = 2)                                              #take mean over all the channels to get a single gradient value
	return gradient


def rgb2gray(rgb):
    	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	sobel_vertical = np.array([[-1, 0, 1],
								[-2, 0, 2],
								[-1, 0, 1]])
	sigma_dog = [3,5]
	kernelsize = 49
	orientations1 = 16
	dog_filters = DOG_filters(sigma_dog, kernelsize, orientations1, sobel_vertical)
	images_plot((20,2), dog_filters, x_len = 16, y_len = 2, name = 'Code/afilters/DoG_filters.png')

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	sigma_lm_small = [1, np.sqrt(2), 2, 2*np.sqrt(2)]
	lm_small_filters = LM_filters(sigma_lm_small, kernelsize)
	images_plot((12,4), lm_small_filters, x_len = 12, y_len = 4, name = 'Code/afilters/LM.png')

	sigma2_large = [np.sqrt(2), 2, 2*np.sqrt(2), 4]
	# lm_large_filters = LMf_ilters(sigma_lm_large, kernelsize)
	# images_plot((20,4), lm_large_filters, x_len = 12, y_len = 4, name = 'LM_large_filters.png')

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	sigma3 = [3,5,7,9,11]
	gabor = gabor_filters(sigma3,kernelsize, theta = 0.25, Lambda = 1, psi = 1, gamma = 1, number = 8)
	images_plot((8,5), gabor, x_len = 8, y_len = 5, name = 'Code/afilters/Gabor.png')

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disc = Halfdisc_Masks(scales = [4, 10, 15])
	images_plot((8,6), half_disc, x_len = 8, y_len = 6, name = 'Code/afilters/HDMasks.png')

	for i in range(1, 11):
		print(i)
		img = image.imread('/home/ankit/Documents/STUDY/RBE594/HW0/Phase1/BSDS500/Images/' + str(i) + '.jpg')
		gray_img = rgb2gray(img)
		maps = []
		grads = []
		comparison = []
		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		texton_m = Texton_map(img, dog_filters, lm_small_filters, gabor, 64)
		pred_t = 3*texton_m
		cm = plt.get_cmap('gist_rainbow')
		color_pred_t = cm(pred_t)
		maps.append(color_pred_t)
		plt.imshow(color_pred_t)
		plt.savefig('Code/texton_maps/TextonMap_' + str(i) + '.png')
		plt.close()
		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		bright_m = Brightness_map(gray_img, 16)
		maps.append(bright_m)
		plt.imshow(bright_m, cmap = 'gray')
		plt.savefig('Code/brightness_maps/BrightnessMap_' + str(i) +'.png')
		plt.close()

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		color_m = color_map(img, 16)
		pred_c = 30*color_m
		color_pred_c = cm(pred_c)
		maps.append(color_pred_c)
		plt.imshow(color_pred_c)
		plt.savefig('Code/color_maps/ColorMap_' + str(i) +'.png')
		plt.close()

		images_plot((12,6), maps, x_len = 3, y_len = 1, name = 'Code/maps/' + str(i) + '.png')

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""		
		texton_gradient = Gradients(texton_m, 64, half_disc)
		grads.append(texton_gradient)
		plt.imshow(texton_gradient)
		plt.savefig('Code/texton_gradients/Tg_' + str(i) + '.png')
		plt.close()

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		brightness_gradient = Gradients(bright_m, 16, half_disc)
		grads.append(brightness_gradient)
		plt.imshow(brightness_gradient)
		plt.savefig('Code/brightness_gradients/Bg_' + str(i) + '.png')
		plt.close()

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		color_gradient = Gradients(color_m, 16, half_disc)
		grads.append(color_gradient)
		plt.imshow(color_gradient)
		plt.savefig('Code/color_gradients/Cg' + str(i) + '.png')
		plt.close()

		#plot all gradients
		images_plot((12,6), grads, x_len = 3, y_len = 1, name = 'Code/gradients/' + str(i) + '.png')

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		img_canny = image.imread('/home/ankit/Documents/STUDY/RBE594/HW0/Phase1/BSDS500/CannyBaseline/' + str(i) + '.png')
		comparison.append(img_canny)

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		img_sobel = image.imread('/home/ankit/Documents/STUDY/RBE594/HW0/Phase1/BSDS500/SobelBaseline/' + str(i) + '.png')
		comparison.append(img_sobel)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		pb_lite = (1/3)*(texton_gradient + brightness_gradient + color_gradient) * (0.5*img_canny + 0.5*img_sobel)
		comparison.append(pb_lite)
		plt.imshow(pb_lite, cmap = 'gray')
		plt.savefig('Code/pb-lite_outputs/PbLite_' + str(i) + '.png')
		plt.close()

		#plot comparisons between canny, sobel and pb_lite
		images_plot((12, 6), comparison, x_len = 3, y_len = 1, name = 'Code/comparison/' + str(i) + '.png')

	   
if __name__ == '__main__':
    main()
 


