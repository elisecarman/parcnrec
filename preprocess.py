import pickle
import numpy as np
import tensorflow as tf
import csv
import gzip
import requests
import os
from PIL import Image
from io import BytesIO
import math
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


"reference to turn images to bytes: https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format/blob/master/convert-images-to-mnist-format.py"

MAX_HEIGHT = 480
MAX_WIDTH = 640

def pad(arr):
	if arr.shape[0] > MAX_HEIGHT:
		print('too high',arr.shape[0])
	if arr.shape[1]>MAX_WIDTH:
		print('too wide',arr.shape[1])
	padding_Y_before = (MAX_HEIGHT-arr.shape[0])/2
	padding_Y_after = math.floor(padding_Y_before)
	if(padding_Y_before - padding_Y_after != 0):
		padding_Y = (padding_Y_after,padding_Y_after+1)
	else:
		padding_Y = (padding_Y_after,padding_Y_after)

	padding_X_before = (MAX_WIDTH-arr.shape[1])/2
	padding_X_after = math.floor(padding_X_before)
	if(padding_X_before - padding_X_after != 0):
		padding_X = (padding_X_after,padding_X_after+1)
	else:
		padding_X = (padding_X_after,padding_X_after)
	#print(padding_Y,padding_X)
	arr = np.pad(arr, [padding_Y, padding_X, (0, 0)], mode='constant')
	return arr

def download_and_pad(url):
	try:
		b = requests.get(url)
		image = Image.open(BytesIO(b.content))
		image = tf.keras.preprocessing.image.img_to_array(image)
		pad_arr = pad(image)
		pad_arr = np.reshape(pad_arr, [640, 480])

		#use max pool

		"""b = requests.get(url)
		img =Image.open(BytesIO(b.content))
		arr = np.asarray(img)
		pad_arr = pad(arr)"""

	except:
		pad_arr = np.zeros((640, 480))
	pad_arr_int = pad_arr.astype(np.int)
	return pad_arr_int

def download_images_from_links(image_array, npy_file_path):
	print("starting downloading")
	with ThreadPoolExecutor(max_workers=14) as executor:
		big_gen = executor.map(download_and_pad, image_array)

	#pad images
	big_list = list(big_gen)

	"bug here: 'MemoryError: Unable to allocate array with shape (10942, 640, 480) and data type float64' "
	big_array = np.array(big_list)

	print("got here")
	np.save(npy_file_path, big_array)

	return None

def early_processing(cat_path,image_path,cat_out_path, img_out_path):
	""" only run ONCE """
	""" rearrange inputs and labels, download images from links """

	id2cat = {}
	id2image = {}
	image2id = {}

	# categories sheet
	with open(cat_path, newline='') as f:
		reader = csv.reader(f)
		for row in reader:
			try:
				id2cat[row[0]].append(row[1])
			except:
				id2cat[row[0]] = [row[1]]

	# image sheet
	with open(image_path, newline='') as f2:
		reader = csv.reader(f2)
		for row in reader:
			image2id[row[1]] = row[0]

		id2image = dict([(value,key)for key,value in image2id.items()])

	# convert two dictionaries to one 2d array
	image2cat = []
	for id in id2image:
		try:
			image2cat.append([id2image[id], id2cat[id]])
		except:
			pass

	# convert one array into two arrays
	altogether = np.array(image2cat).T
	image_array = altogether[0]
	cat_array = altogether[1]

	download_images_from_links(image_array, img_out_path)
#	download_images_from_links(image_array, 'd:/DeepLearning/FinalProj/data/inputs.npy')

	np.save(cat_out_path,cat_array)

	return None


early_processing('d:/DeepLearning/FinalProj/data/categories.csv','d:/DeepLearning/FinalProj/data/images.csv','d:/DeepLearning/FinalProj/data/labels.npy','d:/DeepLearning/FinalProj/data/inputs.npy')


def print_unique(cat_array):
	unique_cats = []
	for row in cat_array:
		for element in row:
			if element not in unique_cats:
				unique_cats.append(element)
	return unique_cats

def sort_labels(cat_array):
	this_cat = []
	for lst in cat_array:
		this_lst = []
		for element in lst:
			if element in ['Art', 'Arts & Crafts', 'Art in the Parks: Celebrating 50 Years', 'Art in the Parks: UNIQLO Park Expressions Grant']:
				element = 1

			if element in ['GreenThumb Events', 'GreenThumb Partner Events', 'GreenThumb 40th Anniversary', 'GreenThumb Workshops']:
				element = 2

			if element in ['Festivals', 'Historic House Trust Festival', "Valentine's Day", 'Halloween', "Saint Patrick's Day", 'Earth Day & Arbor Day', "Mother's Day", "Father's Day", 'Holiday Lightings', "Santa's Coming to Town", 'Lunar New Year', 'Pumpkin Fest', 'Summer Solstice Celebrations', 'Easter', 'Fall Festivals', "New Year's Eve", 'Winter Holidays', 'Thanksgiving', 'National Night Out']:
				element = 3

			if element in ['Volunteer', 'MillionTreesNYC: Volunteer: Tree Stewardship and Care', 'Martin Luther King Jr. Day of Service', 'MillionTreesNYC: Volunteer: Tree Planting']:
				element = 4

			if element in ['Film', 'Free Summer Movies', 'Theater', 'Free Summer Theater', 'Movies Under the Stars', 'Concerts', 'Free Summer Concerts', 'SummerStage']:
				element = 5

			if element in ['Fitness', 'Outdoor Fitness', 'Running', 'Bike Month NYC', 'Hiking', 'Learn To Ride', 'Sports', 'Kayaking and Canoeing', 'National Trails Day', 'Brooklyn Beach Sports Festival', 'Summer Sports Experience', 'Fishing', 'Girls and Women in Sports', 'Bocce Tournament']:
				element = 6

			if element in ['Best for Kids', 'Kids Week', 'CityParks Kids Arts', 'School Break', 'Family Camping', 'CityParks PuppetMobile', 'Dogs']:
				element = 7

			if element in ['Black History Month', "Women's History Month", 'LGBTQ Pride Month', 'Hispanic Heritage Month', 'Native American Heritage Month', 'Fourth of July', 'City of Water Day', "She's On Point"]:
				element = 8

			if element in ['History', 'Historic House Trust Sites', 'Arts, Culture & Fun Series', 'Shakespeare in the Parks']:
				element = 9

			if element in ['Open House New York', 'Community Input Meetings', 'Dogs in Parks: Town Hall', 'Parks Without Borders']:
				element = 10

			if element in ['Nature', 'Birding', 'Wildlife', 'Wildflower Week', 'Cherry Blossom Festivals', 'Waterfront', 'Rockaway Beach', 'Bronx River Greenway', 'Fall Foliage', 'Summer on the Hudson', 'Living With Deer in New York City']:
				element = 11

			if element in ['Talks', 'Education', 'Astronomy', 'Partnerships for Parks Tree Workshops']:
				element = 12

			if element in ['Seniors', 'Accessible']:
				element = 13

			if element in ['Dance', 'Games', 'Recreation Center Open House', 'NYC Parks Senior Games', 'Mobile Recreation Van Event', 'Fireworks', 'Tours', 'Freshkills Tours']:
				element = 14

			if element in ['Markets', 'Food']:
				element = 15

			if element in ['Northern Manhattan Parks', 'Fort Tryon Park Trust', "It's My Park", 'Poe Park Visitor Center', 'Shape Up New York', 'Freshkills Park', 'Freshkills Featured Events', 'Urban Park Rangers', 'City Parks Foundation', 'Reforestation Stewardship', 'Forest Park Trust', 'City Parks Foundation Adults', 'Partnerships for Parks Training and Grant Deadlines', 'My Summer House NYC', 'Community Parks Initiative', 'Anchor Parks']:
				element = 16

			if element not in this_lst:
				"NOTE: THIS WAS ADDED SO WE MAY DO ONE HOT VECTORS"
				this_lst.append(element)

		this_cat.append(this_lst)
	return np.array(this_cat)
""

def get_data(image_path,cat_path):

	cat_array = np.load(cat_path, allow_pickle=True)
	img_array = np.load(image_path, allow_pickle=True)

	this_cat = sort_labels(cat_array)
	"then change this to be a 16 sized one hot vector"
	"reference: https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/"

	# turn labels into one hot vectors
	mlb = MultiLabelBinarizer()
	mlb.fit(this_cat)
	labels = np.arange(1, 17)
	mlb.classes_(labels)
	mlb.transform(this_cat)
	# normalization

	img_array = tf.reshape(img_array, [10942, 640, 480]).astype(np.float32)
	img_array = img_array / 255

	"""	test_size = np.math.floor(.3 * 10942)
	train_size = 10942 - test_size
	
	"""


	(train_img, test_img, train_lab, test_lab) = train_test_split(img_array, this_cat, test_size=0.2, random_state=42)

	""""
	new_features = mlb.fit_transform(features)
	"""


	#return this_cat[0 : train_size], img_array[0 : train_size], this_cat[train_size : 10942], img_array[train_size : 10942]
	return train_img, test_img, train_lab, test_lab

#get_data( "#./data/inputs.npy", "#./data/labels.npy")

#get_data('d:/DeepLearning/FinalProj/data/inputs.npy','d:/DeepLearning/FinalProj/data/labels.npy')
