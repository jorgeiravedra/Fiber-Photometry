#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 11/23/2020

@author: Jorge Iravedra, adapted from Branco Lab fiber photometry pipeline
		 Revised on 01/08/2021
'''

import cv2
import numpy as np 
import csv
import os
import pandas as pd
from time import mktime
from path import Path 
import sys
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import moviepy.editor as moviepy 
import astropy.convolution as conv
import matplotlib.pyplot as plt
from scipy.signal import resample
from utils import get_cap_from_file, get_cap_selected_frame, get_video_params, cap_set_frame, get_analogue_inputs, manually_define_rois, split_channels, smooth_edges, realign_columns

################################################
#			   Class definition				   #
################################################

DEBUG = False
if DEBUG:
	plot_double_exp=False
	plot_correction=False
	plot_dff=False

class SignalExtraction:
	def __init__(self, video_path, n_rois, roi_radius=40, save_path=None, overwrite=False, traces_file=None):

		# Open video and stores a few relevant vars
		self.video_path = video_path

		if 'avi' in video_path: # convert .avi to .mp4
			if os.path.isfile(video_path.split('.')[0]+'.mp4') == True:
				video_path = video_path.split('.')[0]+'.mp4'
			else: 
				clip = moviepy.VideoFileClip(video_path)
				clip.write_videofile(video_path.split('.')[0]+'.mp4')
				video_path = video_path.split('.')[0]+'.mp4'

		self.video = get_cap_from_file(video_path)
		self.video_params = get_video_params(self.video) # nframes, width, height, fps
		self.first_frame = get_cap_selected_frame(self.video, 0)
		self.ref_frame = self.first_frame
		if use_ref_frame == True:
			self.ref_frame = get_cap_selected_frame(self.video, 1) # play around with this to select particular frame
		self.n_rois = n_rois
		self.roi_radius = roi_radius

		if save_path is not None:
			if 'hdf' not in save_path:
				raise ValueError("The save path should point to a hdf file")
			self.save_path = save_path
		else:
			self.save_path = self.video_path.split(".")[0]+"_traces.hdf"
		self.save_raw_path = self.video_path.split(".")[0]+"_raw_traces.hdf"
		self.overwrite = overwrite

		self.traces_file = traces_file
		
		self.ROI_masks = []

	################################################
	#			   Signal extraction   			   #
	################################################

	def get_ROI_masks(self, mode='manual'):

		"""
		Gets the ROIs for a recording as a list of masked numpy arrays.
			:param frame: first frame of the video to analyse
			:param mode: str, manual for manual ROIs identification, else auto
		"""

		# Get ROIs

		ROIs = manually_define_rois(self.ref_frame, self.n_rois, self.roi_radius)

		# Get ROIs masks
		self.ROI_masks = []

		for roi in ROIs:
			blank = np.zeros_like(self.first_frame)
			cv2.circle(blank, (roi[0], roi[1]), self.roi_radius, (255, 255, 255), -1) 
			blank = (blank[:, :, 0]/255).astype(np.float64)
			blank[blank == 0] = np.nan
			self.ROI_masks.append(blank)

	def extract_signal(self):
		"""
			Extract signal from a video of the fiber bundle. It will need to use ROi masks, so run 
			get_ROI_masks first. For each frame and each mask the frame is multiplied by the mask 
			to get just the pixels in the ROI and the average of these is taken as the signal for that
			ROI at that frame. Results are saved as a pandas dataframe (as a .h5 file)
		"""

		if not self.ROI_masks and self.traces_file is None: 
			print("No ROI masks were defined, please run get_ROI_masks first")
			return None

		if self.traces_file is None:
			# Reset cap
			cap_set_frame(self.video, 0)

			# Prepare arrays to hold the data
			traces = {k:np.zeros(self.video_params[0]) for k in range(self.n_rois)}

			# Extract
			if not DEBUG:
				for frame_n in tqdm(range(self.video_params[0])):
					ret, frame = self.video.read()
					if not ret: 
						raise ValueError("Could not read frame {}".format(i))

					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

					for n, mask in enumerate(self.ROI_masks):
						masked = (frame * mask).astype(np.float64)
						traces[n][frame_n] = np.nanmean(masked.astype(np.uint8))

				traces = pd.DataFrame(traces)

				# Split traces
				traces = split_channels(traces)
		else:

			traces = pd.read_hdf(self.traces_file)

		raw_traces = traces.copy()

		print(raw_traces.shape)

		if first_frame == 'green' or first_frame == 'isosbestic': 
		# realigns columns to first index corresponding to blue light frames
			traces = realign_columns(traces, first_frame=first_frame)

		traces = traces.apply(lambda x: x.fillna(x.median()), axis=0) # some frames get 
																	  #	cut off at stop so fill those
		# Remove double exponential
		traces = self.remove_double_exponential(traces)

		# Regress violet from blue. 
		traces = self.regress_isosbestic(traces)

		# Compute DF/F
		traces = self.compute_dff(traces)
	   
		# Save and return
		print("Extraction completed.")
		print("Saving raw traces at: {}".format(self.save_raw_path))
		raw_traces.to_hdf(self.save_raw_path, key='hdf')
		print("Saving  processed traces at: {}".format(self.save_path))
		traces.to_hdf(self.save_path, key='hdf')

		return raw_traces, traces

	################################################
	#			   Postprocessing   			   #
	################################################

	@staticmethod
	def double_exponential(x, a, b, c, d):
		return a * np.exp(b * x) + c * np.exp(d*x)

	def remove_exponential_from_trace(self, x, y):
		""" Fits a double exponential to the data and returns the results
		
			:param x: np.array with time indices
			:param y: np.array with signal
			:returns: np.array with doble exponential corrected out
		"""
		popt, pcov = curve_fit(self.double_exponential, x, y, 
			maxfev=10000, 
			p0=(0.7,  -1e-6, 0.7,  -1e-6), 
			bounds = [[0, -1e-1, 0, -1e-1], [50, 0, 50, 0]])

		fitted_doubleexp = self.double_exponential(x, *popt)
		y_pred = y - (fitted_doubleexp - np.min(fitted_doubleexp))
		return y_pred, fitted_doubleexp

	def remove_double_exponential(self, traces):

		time = np.arange(len(traces))

		for column in traces.columns:
			before = traces[column].values.copy()
			traces[column], double_exp = self.remove_exponential_from_trace(time, traces[column].values)

			if DEBUG and plot_double_exp:
				plt.plot(before, color='green', lw=2, label='raw')
				plt.plot(traces[column].values, color='k', lw=.5, label='correct')
				plt.plot(double_exp, color='red', label='double exp')
				plt.legend()
				plt.show()

		return traces

	def compute_dff(self, traces):

		for column in traces.columns:
			trace = traces[column].values.copy()
			baseline = np.nanmedian(trace)
			traces[column] = smooth_edges((trace-baseline)/baseline, 20)

			if DEBUG and plot_dff:
				f, axarr = plt.subplots(nrows=2)
				axarr[0].plot(trace, color='k', label='trace')
				axarr[0].axhline(baseline, color='m', label='baseline')
				axarr[1].plot(traces[column], color='g', label='dff')
				for ax in axarr: ax.legend()
				plt.show()

		return traces

	def regress_isosbestic(self, traces):

		# we consider odd numbers to be in green channel and evens to be in blue channel

		for roi_n in np.arange(self.n_rois):
			green = traces[str(roi_n)+'_green'].values
			blue = traces[str(roi_n)+'_blue'].values
			violet = traces[str(roi_n)+'_isosbestic'].values

			# signal correction on blue (gcamp) channel

			if roi_n % 2 == 0: # start ROI selection with blue channel

				regressor_blue = LinearRegression()  
				regressor_blue.fit(violet.reshape(-1, 1), blue.reshape(-1, 1))
				expected_blue = regressor_blue.predict(violet.reshape(-1, 1)).ravel()        
				corrected_blue = ((blue - expected_blue)/expected_blue)
				traces[str(roi_n)+'_blueDff'] = corrected_blue

			# signal correction on green (rcamp) channel; but this has to be done using the isosbestic from the blue

			else:

				regressor_green = LinearRegression()  
				regressor_green.fit(violet.reshape(-1, 1), green.reshape(-1, 1))
				expected_green = regressor_green.predict(violet.reshape(-1, 1)).ravel()        
				corrected_green = ((green - expected_green)/expected_green)
				traces[str(roi_n)+'_greenDff'] = corrected_green

			if DEBUG and plot_correction:

				if roi_n % 2 == 0:

					# plot 1
					f, axarr = plt.subplots(nrows=3)
					axarr[0].plot(blue, color='b', label='blue')
					axarr[1].plot(violet, color='m',  label='violet')
					axarr[1].plot(expected_blue, color='k',  label='expected_blue')
					axarr[2].plot(corrected_blue, color='red', label='corrected_blue')
					for ax in axarr: ax.legend()
					plt.show()

				else:
					# plot 2
					f, axarr = plt.subplots(nrows=3)
					axarr[0].plot(green, color='b', label='green')
					axarr[1].plot(violet, color='m',  label='violet')
					axarr[1].plot(expected_green, color='k',  label='expected_green')
					axarr[2].plot(corrected_green, color='green', label='corrected_green')
					for ax in axarr: ax.legend()
					plt.show()

		return traces

################################################
#			 		  Misc   				   #
################################################

	def check_frames(self, videodir, videoname, traces):
		'''
		checks for dropped frames, plots relevant frame data and adds timestamps to traces from DAQ csv file
		this function also corrects for any npm frame misalignment (assuming the problem is the 
		video started off in the wrong frame)
		'''
		print('Plotting frame and ttl information...')
		frames_dir = videodir+'frame_data/'
		if os.path.isdir(frames_dir) == False: 
			os.mkdir(frames_dir)

		samples, time, ttl, npmVolt, _ = get_analogue_inputs(videodir, videoname)

		# calculations

		ttlON, ttlONs = np.argmax(np.diff(ttl)) + 1, time[np.argmax(np.diff(ttl))]
		ttlOFF, ttlOFFs = np.argmin(np.diff(ttl)) + 1, time[np.argmin(np.diff(ttl))]
		idxsNPM = np.where(np.diff(npmVolt) <= -0.705)[0] + 1
		npmTimestamps = time[idxsNPM]
		
		# check for anomaly frames
		unique_idxs = np.unique(np.diff(idxsNPM)).astype(float)
		median_idx = np.median(np.diff(idxsNPM)) # sampling rate of frames
		for i in unique_idxs: # check for idxs where frames are not sampled uniformly 
			if i >= median_idx * 1.5:
				anomaly = np.where(np.diff(idxsNPM)==i)[0]
				print('A total of %i frames were unevenly sampled' % anomaly.size)
				print('Unique frame diff values = ' + str(unique_idxs))
				print('Median frame diff = ' + str(median_idx))

		# some other useful frame data
		npmFrame = len(idxsNPM)
		t_video = ttlOFFs - ttlONs 
		video_DAQdataPts = ttlOFF - ttlON
		npmFs = npmFrame/t_video

		if plot_frameData == True:

			# first pass frame info
			plt.figure(figsize=(15, 4))
			plt.plot(time, npmVolt)
			plt.scatter(time[idxsNPM], npmVolt[idxsNPM], color='red')
			plt.title('NPM (V)')
			plt.xlabel('time (s)')
			plt.tight_layout()
			sns.despine()
			plt.savefig(frames_dir+videoname+'_firstPass.png')
			plt.show()

			plt.figure(figsize=(15, 4))
			plt.plot(time, ttl)
			plt.axvline(ttlONs, linestyle='--', color='r', label='TTL ON')
			plt.axvline(ttlOFFs, linestyle='--', linewidth=2, color='g', label='TTL OFF')
			plt.title('TTL validation')
			plt.xlabel('time (s)')
			plt.tight_layout()
			plt.legend()
			sns.despine()
			plt.savefig(frames_dir+videoname+'_TTLValid.png')
			plt.show()

			plt.figure(figsize=(15, 4))
			plt.plot(np.diff(idxsNPM))
			plt.title('Check for dropped NPM frames')
			
			try:
				for anom in anomaly:
					plt.axvline(anom, linestyle='--', color='r')
					print("Dropped frame: " + str(anom))
			except NameError:
				print("No dropped frames")
				
			plt.ylabel('diff')
			plt.xlabel('frame')
			plt.tight_layout()
			sns.despine()
			plt.savefig(frames_dir+videoname+'_droppedFrames.png')
			plt.show()

		return npmTimestamps

	def add_timestamps(self, traces, npmTimestamps, first_frame='blue'):

		print('Adding timestamps from DAQ...')

		# correct for channel timestamps of != lengths

		if first_frame == 'green':

			# make corrections to blue channel if first frame is green
			blue = npmTimestamps[2::3]
			green = npmTimestamps[0::3]
			violet = npmTimestamps[1::3]

			# blue_dt = np.median(np.diff(blue)) 
			# blue = np.array(pd.concat((pd.Series(blue).shift(periods=1), pd.Series(blue[-1]))))
			# blue[0] = blue[1] - blue_dt

		if first_frame == 'isosbestic':

			# make corrections to blue channel if first frame is violet
			blue = npmTimestamps[1::3]
			green = npmTimestamps[2::3]
			violet = npmTimestamps[0::3]

			# blue_dt = np.median(np.diff(blue)) 
			# blue = np.array(pd.concat((pd.Series(blue).shift(periods=1), pd.Series(blue[-1]))))
			# blue[0] = blue[1] - blue_dt
			# green_dt = np.median(np.diff(green)) 
			# green = np.array(pd.concat((pd.Series(green).shift(periods=1), pd.Series(green[-1]))))
			# green[0] = green[1] - green_dt

		else:

			blue = npmTimestamps[0::3]
			green = npmTimestamps[1::3]
			violet = npmTimestamps[2::3]

		chanl_lens = np.unique([blue.size, green.size, violet.size])

		print('---- Timestamp and traces hdf initial shapes ----')
		print(blue.shape), print(green.shape), print(violet.shape), print(traces.shape)
		tar = np.max(chanl_lens)

		if blue.size != tar: 
			blue = np.insert(blue, -1, blue[-1]+np.median(np.diff(blue)))
		if green.size != tar: 
			green = np.insert(green, -1, green[-1]+np.median(np.diff(green)))
		if violet.size != tar: 
			violet = np.insert(violet, -1, violet[-1]+np.median(np.diff(violet)))

		print('---- Timestamp and traces hdf final shapes ----')
		print(blue.shape), print(green.shape), print(violet.shape), print(traces.shape)

		traces['timestamps_blue'] = blue
		traces['timestamps_green'] = green
		traces['timestamps_isosbestic'] = violet
		traces.to_hdf(self.save_path, key='hdf')
			
		return traces

################################################
#			   Choose your fighter!			   #
################################################

dirname = '//Volumes/falkner/Jorge/Multifiber/cohort1_11232020_start/testday4/80180/'
name = '80180_t4_male_12122020_npm_2020-12-12-181645-0000.avi'
video_path = dirname+name
traces_path = video_path.split('.')[0]+'_raw_traces.hdf' # in case implementation isn't from scratch

################################################
#	   Check conds before implementation       #
################################################

plot_frameData = False # check True if you want to check for missing frames
use_ref_frame = True # check True if you want to select ROIs using a frame different from the first
first_frame = 'green' # check 'isosbestic' or 'green' if mp4 starts with a frame with
					  # coresponding LED; otherwise leave True or 'blue'

################################################
#			 	  Implementation   		       #
################################################

n_rois = 22
se = SignalExtraction(video_path, n_rois=n_rois, traces_file=None, overwrite=True)
se.get_ROI_masks()
raw, traces = se.extract_signal()
npmTimestamps = se.check_frames(dirname, name, traces)
traces = se.add_timestamps(traces, npmTimestamps, first_frame=first_frame)

