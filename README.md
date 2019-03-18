# FaceRecognition
training.py
	usage: train.py [-h] [--img_dir IMG_DIR] [--protocol_dir PROTOCOL_DIR]
					[--model_dir MODEL_DIR] [--workers WORKERS]
					[--batch_size BATCH_SIZE] [--split_n SPLIT_N]

	IJB-A dataset Training.
	
	optional arguments:
	-h, --help        	              show this help message and exit
	--img_dir IMG_DIR                 Path to the data directory containing aligned face images.
	--protocol_dir PROTOCOL_DIR       Path to the data directory containing csv files for metadata and template pairs.
	--model_dir MODEL_DIR             Path to directory to save trained models.
	--workers WORKERS                 Number of workers for data loader.
	--batch_size BATCH_SIZE           Number of batches for data loader to divide data into batches.
	--split_n SPLIT_N                 Split NUmber for which model is to be trained.

	Example : python train.py --img_dir C:/Data/Assignment2/ijba_aligned_all --protocol_dir C:/Data/Assignment2/IJB-A_11_sets --model_dir Models --workers 0 --batch_size 64 --split_n 5
verify.py
	usage: verify.py [-h] [--img_dir IMG_DIR] [--protocol_dir PROTOCOL_DIR]
					[--model_dir MODEL_DIR] [--features_dir FEATURES_DIR]
					[--workers WORKERS] [--batch_size BATCH_SIZE]
					[--split_n SPLIT_N] [--extract_features] [--roc]
	
	IJB-A Protocol 1:1 verification.
	
	optional arguments:
	-h, --help        	              show this help message and exit
	--img_dir IMG_DIR                 Path to directory containing aligned face images.
	--protocol_dir PROTOCOL_DIR       Path to directory containing csv files for metadata and template pairs.
	--model_dir MODEL_DIR             Path to directory containing trained models.
	--features_dir FEATURES_DIR       Path to directory to store and load image feature vectors.
	--workers WORKERS                 Number of workers for data loader.
	--batch_size BATCH_SIZE           Number of batches for data loader to divide data into batches.
	--split_n SPLIT_N                 Split Number for which features are extracted. If absent, all splits will be processed
	--extract_features                Flag to extract features of images from trained model
	--roc                             Flag to plot ROC curves using extracted features
	
	Example : python verify.py --img_dir C:/Data/Assignment2/ijba_aligned_all --protocol_dir C:/Data/Assignment2/IJB-A_11_sets --model_dir Models --features_dir Features --workers 0 --batch_size 64 --roc --extract_features --split_n 5
