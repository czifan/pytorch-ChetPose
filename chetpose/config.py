import os

class Config:
	data_dir = './data/lsp_data'
	train_json_file = os.path.join(data_dir, 'train.json')
	train_img_dir = os.path.join(data_dir, 'train')
	val_json_file = os.path.join(data_dir, 'test.json')
	val_img_dir = os.path.join(data_dir, 'test')


	cuda = True
	if cuda:
		num_workers = 6
	else:
		num_workers = 0
	model_type = 'darknet-base'
	pre_trained = True
	model_path = './checkpoint/darknet-base_best.pth.tar'

	num_kpt = 14
	size = 224
	s = 7
	max_degree = 40
	ratio_max_x = 0.1
	ratio_max_y = 0.1
	center_perturb_max = 20
	prob = 0.5
	min_gauss = 2
	max_gauss = 4
	min_alpha = 0.5
	max_alpha = 2.5
	percentage = 0.1
	batch_size = 24

	start_epoch = 1
	end_epoch = 201

	threshold = 0.2
	gamma = 0.995
	learning_rate = 0.00004
	lr_gamma = 0.5
	weight_decay = 0.00001
	max_pck = 0.0
	display = 80
	evaluation = 1

	checkpoint = 'checkpoint/'
	filename = '_best.pth.tar'

	decay = [20, 40, 60, 80, 100, 120, 140, 160, 180]

	# LSP PCK 
	lsp_pck_thres = 0.2

	# visualize
	limbSeq = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [12, 14], [14, 15]]
	colors = [[1., 0., 0.], [1., 0.33, 0], [1., 0.66, 0], [1., 1., 0], [0.66, 1., 0], [0.33, 1., 0], [0, 1., 0], \
	          [0, 1., 0.33], [0, 1., 0.66], [0, 1., 1.], [0, 0.66, 1.], [0, 0.33, 1.], [0, 0, 1.], [0.33, 0, 1.], \
	          [0.66, 0, 1.], [1., 0, 1.], [1., 0, 0.66], [1., 0, 0.33]]


config = Config()