#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def data_check(dirname): 
	os.chdir(dirname)
	cwd = os.getcwd()
	dirlist = os.listdir('.')

	# check hdf files available (for raw files simply insert '_raw' before 'traces')
	print('----- HDF FILES AVAILABLE -----')
	for file in dirlist:
		if '.hdf' in file:
			if '.png' not in file:
				if 'raw' not in file:
					print(file)

	print('------ EXTRACTABLE FILES ------')
	for file in dirlist:
		if 'traces' not in file:
			if 'avi' in file:
				if 'npm' in file: 
					print(file)

	print('--------- FRAME DATA? ---------')
	dirname = './frame_data'
	flist = []
	if os.path.isdir(dirname) == True:
		for filename in os.listdir(dirname):
			temp = filename
			flist.append(filename.split('_')[2])
	print(np.unique(flist))

def troubleshoot_raws(filename):
	'''
	Takes a filename from raw hdf and plots raw traces for each ROI type
	ROI ORDER IS IMPORTANT (make sure blue ROIs are even and green ROIs odd!)
	'''
	raw_data = pd.read_hdf(filename)

	fig, ax = plt.subplots(11, figsize=(8,30))

	for column in raw_data.columns:
		num = int(column.split('_')[0])
		if num % 2 == 0: # check for GCaMP ROIs
			idx = int(num/2)
			ax[idx].plot(raw_data[column])
			sns.despine()

	ax[0].legend(['GCaMP', 'RCaMP', 'Isos'])
	ax[0].set_title('GCaMP ROIs')
	ax[-1].set_xlabel('Frames')
	plt.savefig(filename+'_rcamp_troubleshooting_ttlONBlue.png')
	
	plt.close()
	
	fig, ax = plt.subplots(11, figsize=(8,30))

	for column in raw_data.columns:
		num = int(column.split('_')[0])
		if num % 2 != 0: # check for RCaMP ROIs
			idx = int(num/2)
			ax[idx].plot(raw_data[column])
			sns.despine()

	ax[0].legend(['GCaMP', 'RCaMP', 'Isos'])
	ax[0].set_title('RCaMP ROIs')
	ax[-1].set_xlabel('Frames')
	plt.savefig(filename+'_rcamp_troubleshooting_ttlONGreen.png')

def plot_dff(filename):
	
	# ensure corresponding directories
	
	dirpath = './figures/'+filename.split('_')[2]+'/'
	
	try:
		if not os.path.exists(os.path.dirname('./figures')):
			os.makedirs(os.path.dirname('./figures'))
	except OSError as err:
		print(err)
		
	try:
		if not os.path.exists(os.path.dirname(dirpath)):
			os.makedirs(os.path.dirname(dirpath))
	except OSError as err:
		print(err)
	
	raw_data = pd.read_hdf(filename)
	blue_t = raw_data['timestamps_blue']
	green_t = raw_data['timestamps_green']
	viol_t = raw_data['timestamps_isosbestic']
	
	dff = pd.DataFrame()
	for col in raw_data.columns:
		if 'Dff' in col:
			dff = pd.concat([dff, raw_data[col]], axis=1)
	
	# aesthetics
	
	colors = sns.color_palette("coolwarm")
			
	# plot everything
			
	for i, col in enumerate(dff.columns):
		
		if 'blue' in col:
			
			fig, ax = plt.subplots(1, 2, figsize=(10, 2))
			
			# plot raw data
			ax[0].plot(viol_t, raw_data[col.split('_')[0]+'_isosbestic'], color='purple', label='isosbestic')
			ax[0].plot(blue_t, raw_data[col.split('_')[0]+'_blue'], color=colors[0], label='blue channel', alpha=0.5)
			ax[0].set_ylim(top=0.1, bottom=-0.1)
			ax[0].set_ylabel('Raw F')

			# plot dff
			ax[1].plot(blue_t, -dff[col], color=colors[0], label='corrected blue channel')
			ax[1].set_ylabel('\u0394 F/F')
			
			plt.tight_layout()
			sns.despine()
			
			# save fig to figures directory
			plt.savefig(dirpath+'_'.join(filename.split('_')[:3])+'_ROI%i.png' % i, dpi=500)

		if 'green' in col:
			
			fig, ax = plt.subplots(1, 2, figsize=(10, 2))
			
			# plot raw data
			ax[0].plot(viol_t, raw_data[col.split('_')[0]+'_isosbestic'], color='purple', label='isosbestic')
			ax[0].plot(green_t, raw_data[col.split('_')[0]+'_green'], color=colors[-1], label='green channel', alpha=0.5)
			ax[0].set_ylim(top=0.1, bottom=-0.1)
			ax[0].set_ylabel('Raw F')

			# plot dff
			ax[1].plot(green_t, -dff[col], color=colors[-1], label='corrected green channel')
			ax[1].set_ylabel('\u0394 F/F')
			
			plt.tight_layout()
			sns.despine()
			
			# save fig to figures directory
			plt.savefig(dirpath+'_'.join(filename.split('_')[:3])+'_ROI%i.png' % i, dpi=500)

######## CHECK DIRECTORY FOR FILES TO BE PLOTTED ########

path = '//Volumes/falkner/Jorge/Multifiber/cohort1_11232020_start/testday4/80180/'
os.chdir(path)
data_check('.')

################# FOR TROUBLESHOOTING ###################


