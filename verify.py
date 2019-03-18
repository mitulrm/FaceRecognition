# Name : Mitul Modi
# mitulraj@buffalo.edu

import sys
import os
import csv
import time
import PIL
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from utils import *

def main(args):
	img_dir = os.path.expanduser(args.img_dir) if args.img_dir is not None else os.getcwd()
	protocol_dir = os.path.expanduser(args.protocol_dir) if args.protocol_dir is not None else os.getcwd()
	model_dir = os.path.expanduser(args.model_dir) if args.model_dir is not None else os.getcwd()
	features_dir = os.path.expanduser(args.features_dir) if args.features_dir is not None else os.getcwd()
	workers = args.workers if args.workers is not None else 0
	batch_size = args.batch_size if args.batch_size is not None else 64

	classes_n = [332, 332, 333, 333, 333, 333, 333, 333, 333, 333]
	transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

	# Check if GPU is available
	gpu = torch.cuda.is_available()
	if gpu:
		print('GPU is available.')
	else:
		print('GPU not available. Executing on CPU.')
		
	# If split_n argument is passed, process only specified split else process all splits.
	if args.split_n is None:
		split_start = 1
		split_end = 11
	else:
		split_start = args.split_n
		split_end = args.split_n + 1


	if args.extract_features:
			
		for split_n in range(split_start, split_end):

			print('Extracting features for Split ', split_n)
			metadata_df = get_metadata(protocol_dir, img_dir, split_n)

			# Define dataloader for verification images
			dataloader_test = torch.utils.data.DataLoader(Dataset_Test(np.array(metadata_df['FILE']), transform), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory = True, drop_last = True)

			# Define and load trained model
			model = models.resnet18(pretrained = False)
			features_n = model.fc.in_features
			model.fc = nn.Linear(features_n, classes_n[split_n])

			model = load_model(os.path.join(model_dir, 'best_model_resnet18_'+str(split_n)+'_temp.pth'), model, mode = 'test')

			# Extract features of all verification images using trained model
			image_features = extract_features(model, dataloader_test)

			# Save image features to disk
			np.save(os.path.join(features_dir, 'image_features_split' + str(split_n) + '.npy'), image_features)

	far_mean = np.linspace(0.0001, 1.0, 3000)
	tars = np.zeros((10,3000))
	aucs = np.zeros(10)

	if args.roc:

		for split_n in range(split_start, split_end):
			print('Calculating ROC for Split ', split_n)
			metadata_df = get_metadata(protocol_dir, img_dir, split_n)
			pairs_df = get_template_pairs(protocol_dir, img_dir, split_n)
			image_features = np.load(os.path.join(features_dir, 'image_features_split' + str(split_n) + '.npy'))

			metadata_df = metadata_df[:image_features.shape[0]]
			metadata_df['FEATURES'] = list(image_features)  

			# Aggregate feature vectors of all images with same template Id
			metadata_agg_df = metadata_df.groupby('TEMPLATE_ID').agg({'SUBJECT_ID':'max','FEATURES':lambda x:np.array(x).mean().tolist()}).reset_index()

			# Associate template features and subject Id with corresponding teamplte Id paris for verification.
			pairs_df = pd.merge(pairs_df, metadata_agg_df, left_on = 'TEMPLATE_ID1', right_on = 'TEMPLATE_ID')[['TEMPLATE_ID1', 'TEMPLATE_ID2', 'SUBJECT_ID', 'FEATURES']]
			pairs_df = pd.merge(pairs_df, metadata_agg_df, left_on = 'TEMPLATE_ID2', right_on = 'TEMPLATE_ID', suffixes = ('1','2'))[['TEMPLATE_ID1', 'TEMPLATE_ID2', 'SUBJECT_ID1', 'FEATURES1', 'SUBJECT_ID2', 'FEATURES2']]

			# Calculate groun truth values for each template pair
			pairs_df['GROUND_TRUTH'] = pairs_df['SUBJECT_ID1'] == pairs_df['SUBJECT_ID2']    

			# Find cosine similarity for each template pair
			x = np.array(pairs_df['FEATURES1'].apply(lambda x : np.array(x)))
			x = np.array(x.tolist())
			y = np.array(pairs_df['FEATURES2'].apply(lambda x : np.array(x)))
			y = np.array(y.tolist())

			cs = cosine_similarity(x,y)
			pairs_df['COSINE_SIMILARITY'] = np.diag(cs)   

			# Find True Acceptance Rate, False Acceptance Ratesand AUC value.
			far, tar, thresholds  = roc_curve(pairs_df['GROUND_TRUTH'], pairs_df['COSINE_SIMILARITY'])

			roc_auc = auc(far,tar)

			# Plot ROC Curve
			plot_roc(far, tar, roc_auc, split_n)

			tars[split_n-1] = np.interp(far_mean, far, tar)
			aucs[split_n-1] = roc_auc

		# Plot Avg ROC curve only if ROC for all splits is calculated.
		if args.split_n is None:
			print('Calculating Avg ROC ')
			tar_mean = tars.mean(axis = 0)
			auc_mean = aucs.mean()
			plot_roc(far_mean, tar_mean, auc_mean, 'Avg')

def parse_arguments(argv):
	parser = argparse.ArgumentParser(description='IJB-A Protocol 1:1 verification.')

	parser.add_argument('--img_dir', type = str, help = 'Path to directory containing aligned face images.')
	parser.add_argument('--protocol_dir', type = str, help = 'Path to directory containing csv files for metadata and template pairs.')
	parser.add_argument('--model_dir', type = str, help = 'Path to directory containing trained models.')
	parser.add_argument('--features_dir', type = str, help = 'Path to directory to store and load image feature vectors.')
	parser.add_argument('--workers', type = int, help = 'Number of workers for data loader.')
	parser.add_argument('--batch_size', type = int, help = 'Number of batches for data loader to divide data into batches.')
	parser.add_argument('--split_n', type = int, help = 'Split Number for which features are extracted. If absent, all splits will be processed')
	parser.add_argument('--extract_features', action = 'store_true', help = 'Flag to extract features of images from trained model')
	parser.add_argument('--roc', action = 'store_true', help = 'Flag to plot ROC curves using extracted features')

	return parser.parse_args()

if __name__ == "__main__":
	main(parse_arguments(sys.argv[1:]))