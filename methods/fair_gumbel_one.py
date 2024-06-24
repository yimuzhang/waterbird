import numpy as np
import torch
import torch.optim as optim 
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
from data.utils import MultiEnvDataset
from methods.modules import *


def print_prob(x):
	x = np.squeeze(x)
	out_str = '['
	for i in range(np.shape(x)[0]):
		if i > 0:
			out_str += ' '
		out_str += '%.2f,' % (x[i])
	out_str += ']'
	return out_str


def fair_ll_sgd_gumbel_uni(features, responses, hyper_gamma=10, learning_rate=1e-3, niters=50000, niters_d=2, niters_g=1, offset=-1,
						batch_size=32, mask=None, init_temp=0.5, final_temp=0.05, iter_save=100, log=False):
	'''
		Implementation of FAIR-LL estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		hyper_gamma : float
			hyper-parameter gamma control the degree of invariance
		learning_rate : float
			learning rate for stochastic gradient descent
		niters : int
			number of outer iterations
		niters_d : int
			number of inner iterations for discriminator
		niters_g : int
			number of inner iterations for generator
		batch_size : int
			batch_size for stochastic gradient descent
		init_temp : float
			initial temperature for gumbel approximation
		final_temp: float
			final temperature for gumbel approximation
		log : bool
			whether to show logs during training

		Returns
		----------
		a dict collecting things of interests
	'''
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR Linear/Linear Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	model = FairLinear(dim_x, num_envs)
	model_var = GumbelGate(dim_x, init_offset=offset, device='cpu')
	optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)

	optimizer_g = optim.Adam(model.params_g(), lr=learning_rate)
	optimizer_f = optim.Adam(model.params_f(), lr=learning_rate)

	# construct dataset from numpy array
	dataset = MultiEnvDataset(features, responses)
	gate_rec = []
	weight_rec = []
	loss_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	tau = init_temp
	for it in it_gen:
		# calculate the temperature
		if (it + 1) % 100 == 0:
			tau = max(final_temp, tau * 0.993)
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			optimizer_g.zero_grad()
			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau)).detach()

			out_gs, out_fs = model([gate * x for x in xs])
			loss_de = - sum([torch.mean((ys[e] - out_gs[e].detach()) * out_fs[e] - 0.5 * out_fs[e] * out_fs[e]) for e in range(num_envs)])
			loss_de.backward()
			optimizer_f.step()

		my_loss = np.zeros((niters_g, 2))

		# train the generator
		for i in range(niters_g):
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()

			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau))
			out_gs, out_fs = model([gate * x for x in xs])

			loss_r = 0.5 * sum([torch.mean((out_gs[e] - ys[e]) ** 2) for e in range(num_envs)])
			loss_j = sum([torch.mean((ys[e] - out_gs[e]) * out_fs[e] - 0.5 * out_fs[e] * out_fs[e]) for e in range(num_envs)])
			loss = loss_r + hyper_gamma * loss_j
			loss.backward()
			my_loss[i, 0], my_loss[i, 1] = loss_r.item(), loss_j.item()

			optimizer_g.step()
			optimizer_var.step()

		# save the weight/logits for linear model
		if it % iter_save == 0:
			with torch.no_grad():
				weight = model.g.weight.detach().cpu()
				logits = model_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits))
				weight_rec.append(np.squeeze(weight.numpy() + 0.0))
			#print(logits, np.squeeze(weight.numpy() + 0.0))
			loss_rec.append(np.mean(my_loss, 0))
			if log and it % 20000 == 0:
				print(f'gate = {sigmoid(logits)}')


	ret = {'weight': weight_rec[-1] * sigmoid(logits),
			'weight_rec': np.array(weight_rec),
			'gate_rec': np.array(gate_rec),
			'model': model,
			'fair_var': model_var,
			'loss_rec': np.array(loss_rec)}

	return ret


def fair_ll_classification_sgd_gumbel_uni(features, responses, eval_data=None, hyper_gamma=30, learning_rate=1e-2, niters=50000, niters_d=5, niters_g=1, 
						anneal_rate=0.993, offset=-3, batch_size=32, mask=None, init_temp=0.5, final_temp=0.05, iter_save=100, log=False):
	'''
		Implementation of FAIR-LL estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		hyper_gamma : float
			hyper-parameter gamma control the degree of invariance
		learning_rate : float
			learning rate for stochastic gradient descent
		niters : int
			number of outer iterations
		niters_d : int
			number of inner iterations for discriminator
		niters_g : int
			number of inner iterations for generator
		batch_size : int
			batch_size for stochastic gradient descent
		init_temp : float
			initial temperature for gumbel approximation
		final_temp: float
			final temperature for gumbel approximation
		log : bool
			whether to show logs during training

		Returns
		----------
		a dict collecting things of interests
	'''
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR Linear/Linear Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	model = FairNN(dim_x, 0, 0, 0, 0, num_envs, None)
	model_var = GumbelGate(dim_x, init_offset=offset, device='cpu')
	optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)

	optimizer_g = optim.Adam(model.params_g(), lr=learning_rate, weight_decay=1e-3)
	optimizer_f = optim.Adam(model.params_f(), lr=learning_rate, weight_decay=1e-3)

	# construct dataset from numpy array
	dataset = MultiEnvDataset(features, responses)
	gate_rec = []
	weight_rec = []
	loss_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	if eval_data is not None:
		test_x, test_y = eval_data
		test_x_th = torch.tensor(test_x).float()
		eval_iter = niters // 50
	else:
		eval_iter = niters + 2
	
	tau = init_temp
	loss_rec, acc_rec = [], []
	for it in it_gen:
		# calculate the temperature
		if (it + 1) % 100 == 0:
			tau = max(final_temp, tau * anneal_rate)
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			optimizer_g.zero_grad()
			model.train()

			xs, ys = dataset.next_batch(batch_size)
			
			cat_y = torch.cat(ys, 0)
			cat_y=cat_y.reshape(-1,1)

			gate = model_var.generate_mask((tau_logits, tau)).detach()

			out_g, out_f = model([gate * x for x in xs])
			loss_de = - torch.mean((cat_y - torch.sigmoid(out_g).detach()) * out_f - 0.5 * out_f * out_f)###
			loss_de.backward()
			optimizer_f.step()


		# train the generator
		for i in range(niters_g):
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			model.train()

			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau))
			cat_y = torch.cat(ys, 0)
			cat_y=cat_y.reshape(-1,1)
			out_g, out_f = model([gate * x for x in xs])
			out_prob = torch.sigmoid(out_g)
      
			loss_r = -0.5 * torch.mean((cat_y * torch.log(out_prob + 1e-9) + (1 - cat_y) * torch.log(1 - out_prob + 1e-9)))
			loss_j = torch.mean((cat_y - out_prob) * out_f - 0.5 * out_f * out_f)
			loss = loss_r + hyper_gamma * loss_j
			accuracy = torch.mean((out_g >= 0) * cat_y + (out_g < 0) * (1 - cat_y))
			loss.backward()

			loss_rec.append(loss_r.item())
			acc_rec.append(accuracy.item())
			optimizer_g.step()
			optimizer_var.step()

		# save the weight/logits for linear model
		if it % iter_save == 0:
			with torch.no_grad():
				weight = model.g.relu_stack.weight.detach().cpu()
				logits = model_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits))
				weight_rec.append(np.squeeze(weight.numpy() + 0.0))
			#if log:
			#	print(f'Train Loss = {np.mean(loss_rec)}, Train accuracy = {np.mean(acc_rec)}')
			#	print(f'Gate = {np.sigmoid(logits)}')

		if (it + 1) % eval_iter == 0:
			model.eval()
			with torch.no_grad():
				pred_test = model(torch.sigmoid(model_var.logits) * test_x_th, pred=True).detach().cpu().numpy()

			def accuracy(x, y):
				y=y.reshape(-1,1)
				return np.mean((x >= 0) * y + (x < 0) * (1 - y))
			test_loss = accuracy(pred_test, test_y)

			if log:
				gate = sigmoid(model_var.get_logits_numpy())
        
				print(f'iter = {it}, train acc = {np.mean(acc_rec)}, test acc = {test_loss}, gate = {np.where(gate>0.8)}')
				print(f'gate min = {np.min(gate)}, gate max = {np.max(gate)}')


	ret = {'weight': weight_rec[-1] * sigmoid(logits),
			'weight_rec': np.array(weight_rec),
			'gate_rec': np.array(gate_rec),
			'model': model,
			'fair_var': model_var,}

	return ret


def fairnn_sgd_gumbel_uni(features, responses, eval_data=None, depth_g=2, width_g=128, depth_f=3, width_f=196, offset=-3, anneal_rate=0.993,
						hyper_gamma=100, learning_rate=1e-4, niters=1000, niters_d=5, niters_g=1, weight_decay_f=1e-3, weight_decay_g=1e-3, add_bn=False,
						batch_size=128, mask=None, init_temp=0.5, final_temp=0.05, iter_save=100, gate_samples=20, log=False):
	'''
		Implementation of FAIR-LL estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		hyper_gamma : float
			hyper-parameter gamma control the degree of invariance
		learning_rate : float
			learning rate for stochastic gradient descent
		niters : int
			number of outer iterations
		niters_d : int
			number of inner iterations for discriminator
		niters_g : int
			number of inner iterations for generator
		batch_size : int
			batch_size for stochastic gradient descent
		init_temp : float
			initial temperature for gumbel approximation
		final_temp: float
			final temperature for gumbel approximation
		log : bool
			whether to show logs during training

		Returns
		----------
		a dict collecting things of interests
	'''
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR NN Model Gumbel: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

	model = FairNN(dim_x, depth_g, width_g, depth_f, width_f, num_envs, features, add_bn)

	model_var = GumbelGate(dim_x, init_offset=offset, device='cpu')
	optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)

	optimizer_g = optim.Adam(model.params_g(log), lr=learning_rate, weight_decay=weight_decay_g)
	optimizer_f = optim.Adam(model.params_f(log), lr=learning_rate, weight_decay=weight_decay_f)

	# construct dataset from numpy array
	dataset = MultiEnvDataset(features, responses)
	
	if eval_data is not None:
		test_x, test_y = eval_data
		if isinstance(test_x, list):
			test_x_ths = [torch.tensor(x).float() for x in test_x]
		else:
			test_x_ths = torch.tensor(test_x).float()
		eval_iter = niters // 2
	else:
		eval_iter = niters + 2

	if not log:
		eval_iter = niters

	gate_rec = []
	weight_rec = []
	loss_rec = []
	# start training
	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	tau = init_temp
	for it in it_gen:
		# calculate the temperature
		if (it + 1) % 100 == 0:
			tau = max(final_temp, tau * anneal_rate)
		tau_logits = 1

		# train the discriminator
		for i in range(niters_d):
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			optimizer_g.zero_grad()
			model.train()

			xs, ys = dataset.next_batch(batch_size)
			cat_y = torch.cat(ys, 0)
			gate = model_var.generate_mask((tau_logits, tau)).detach()

			out_g, out_f = model([gate * x for x in xs])
			loss_de = - torch.mean((cat_y - out_g.detach()) * out_f - 0.5 * out_f * out_f)
			loss_de.backward()
			optimizer_f.step()

		my_loss = np.zeros((niters_g, 2))

		# train the generator
		for i in range(niters_g):
			optimizer_g.zero_grad()
			optimizer_var.zero_grad()
			optimizer_f.zero_grad()
			model.train()

			xs, ys = dataset.next_batch(batch_size)
			gate = model_var.generate_mask((tau_logits, tau))
			cat_y = torch.cat(ys, 0)
			out_g, out_f = model([gate * x for x in xs])
			loss_r = 0.5 * torch.mean((out_g - cat_y) ** 2)
			loss_j = torch.mean((cat_y - out_g) * out_f - 0.5 * out_f * out_f)
			loss = loss_r + hyper_gamma * loss_j
			loss.backward()
			my_loss[i, 0], my_loss[i, 1] = loss_r.item(), loss_j.item()

			optimizer_g.step()
			optimizer_var.step()
		if(it%100==0):
			print("iter",it,my_loss[0,0])

		# save the weight/logits for linear model
		if it % iter_save == 0:
			with torch.no_grad():
				logits = model_var.get_logits_numpy()
				gate_rec.append(sigmoid(logits))
			#if log:
			#	print(logits)
			# loss_rec.append(np.mean(my_loss, 0))

		if (it + 1) % eval_iter == 0:
			preds = []
			model.eval()	
			test_loss = []
			# calculate test loss
			acc_num=0
			for e in range(len(test_x_ths)):
				preds = []
				gates = []
				for k in range(gate_samples):
					gate = model_var.generate_mask((tau_logits, tau))
					pred = model(gate * test_x_ths[e], pred=True)
					preds.append(pred.detach().cpu().numpy())
					gates.append((gate).detach().cpu().numpy() + 0.0)
				out = sum(preds) / len(preds)
				test_loss.append(np.mean(np.square(out - test_y[e])))
				if (abs(out - test_y[e])<0.5):
					acc_num=acc_num+1
			if log:
				print(f'iter = {it}, test_loss = {sum(test_loss)/len(test_loss)}\n acc_num = {acc_num/len(test_loss)}')

	ret = {'gate_rec': np.array(gate_rec),
			'model': model,
			'fair_var': model_var,
			'loss_rec': np.array(loss_rec)}

	return ret



def fairnn_sgd_gumbel_refit(features, responses, mask, eval_data, depth_g=1, width_g=128,
						learning_rate=1e-3, niters=50000, weight_decay_g=5e-4,
						batch_size=32, log=False):
	'''
		Implementation of FAIR-LL estimator with gumbel discrete approximation

		Parameter
		----------
		features : list 
			list of numpy matrices with shape (n_k, p) representing the explanatory variables
		responses : list
			list of numpy matrices with shape (n_k, 1) representing the response variable
		hyper_gamma : float
			hyper-parameter gamma control the degree of invariance
		learning_rate : float
			learning rate for stochastic gradient descent
		niters : int
			number of outer iterations
		niters_d : int
			number of inner iterations for discriminator
		niters_g : int
			number of inner iterations for generator
		batch_size : int
			batch_size for stochastic gradient descent
		init_temp : float
			initial temperature for gumbel approximation
		final_temp: float
			final temperature for gumbel approximation
		log : bool
			whether to show logs during training

		Returns
		----------
		a dict collecting things of interests
	'''
	num_envs = len(features)
	dim_x = np.shape(features[0])[1]

	if log:
		print(f'================================================================================')
		print(f'================================================================================')
		print(f'==')
		print(f'==  FAIR NN Model Gumbel Refit: num of envs = {num_envs}, x dim = {dim_x}')
		print(f'==')
		print(f'================================================================================')
		print(f'================================================================================')

		print(f'Gate = {mask}')

	model = FairNN(dim_x, depth_g, width_g, 0, 0, num_envs, features)
	model_var = FixedGate(dim_x, mask, device='cpu')

	optimizer_g = optim.Adam(model.params_g(log), lr=learning_rate, weight_decay=weight_decay_g)

	# construct dataset from numpy array
	dataset = MultiEnvDataset(features, responses)
	
	test_x, test_y = eval_data

	if isinstance(test_x, list):
		test_x_ths = [torch.tensor(x).float() for x in test_x]
	else:
		test_x_ths = torch.tensor(test_x).float()

	eval_iter = niters // 20
	
	loss_rec = []

	if log:
		it_gen = tqdm(range(niters))
	else:
		it_gen = range(niters)

	for it in it_gen:
		# train the generator
		optimizer_g.zero_grad()
		model.train()

		xs, ys = dataset.next_batch(batch_size)
		gate = model_var.generate_mask()
		cat_y = torch.cat(ys, 0)
		out_g, out_f = model([gate * x for x in xs])

		loss = 0.5 * torch.mean((out_g - cat_y) ** 2)
		loss.backward()


		optimizer_g.step()

    

		if (it + 1) % eval_iter == 0:
			preds = []
			model.eval()
			gate = model_var.generate_mask()


			# calculate test loss
			test_loss = []
			gate = model_var.generate_mask()
			for e in range(len(test_x_ths)):
				out = model(gate * test_x_ths[e], pred=True).detach().cpu().numpy()
				test_loss.append(np.mean(np.square(out - test_y[e])))

			if log:
				print(f'iter = {it}, valid_loss = {valid_loss}, test_loss = {test_loss}')

	ret = {'model': model, 'loss_rec': np.array(loss_rec)}

	return ret
