#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np 
import csv
import pandas as pd
import dateutil.parser
from time import mktime
from path import Path 
import sys
from nptdms import TdmsFile
import os
from tqdm import tqdm
from scipy.signal import resample
import moviepy.editor as moviepy 

################################################
#			          Video       			   #
################################################

def cap_set_frame(cap, frame_number):
	"""
		Sets an opencv video capture object to a specific frame
	"""
	cap.set(1, frame_number)

def get_cap_selected_frame(cap, show_frame):
	""" 
		Gets a frame from an opencv video capture object to a specific frame
	"""
	cap_set_frame(cap, show_frame)
	ret, frame = cap.read()

	if not ret:
		return None
	else:
		return frame


def get_video_params(cap):
	""" 
		Gets video parameters from an opencv video capture object
	"""
	if isinstance(cap, str):
		cap = cv2.VideoCapture(cap)

	frame = get_cap_selected_frame(cap, 0)
	
	if frame is None:
		raise ValueError("Could not read frame from cap while getting video params")
	
	if frame.shape[1] == 3:
		is_color = True
	else: is_color = False
	cap_set_frame(cap, 0)

	nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	return nframes, width, height, fps, is_color


def get_cap_from_file(videopath):
	"""
		Opens a video file as an opencv video capture
	"""
	try:
		cap = cv2.VideoCapture(videopath)
	except Exception as e:
		raise ValueError("Could not open video at: " + videopath + f"\n {e}")

	ret, frame = cap.read()
	if not ret:
		raise ValueError("Something went wrong, can't read form video: " + videopath)
	else:
		cap_set_frame(cap, 0)
	return cap

################################################
#			      Analogue inputs       	   #
################################################

def get_analogue_inputs(videodir, videoname):
	
	# locate csv folder with inputs
	for obj in os.listdir(videodir+'/'):
		if os.path.isdir(os.path.join(videodir, obj)):
			if '_'+videoname.split('_')[2] in os.path.join(videodir, obj):
				tarpath = videodir+'/'+obj
			
	analogue_filepath = tarpath+'/'+os.listdir(tarpath)[0]     
	df = pd.read_csv(analogue_filepath, skiprows=6, sep=',')

	# read info off analogue csv

	daqFs = 350 # Hz 

	samples = np.array(df['Sample'])
	time = samples/daqFs 
	ttl = np.array(df['TTL (V)'])
	npmVolt = np.array(df['NPM_capture (V)'])
	flirVolt = np.array(df['spinview_capture (V)'])

	return samples, time, ttl, npmVolt, flirVolt

################################################
#			   ROI selection    			   #
################################################

def manually_define_rois(frame, n_rois, radius):
	"""
		Lets user manually define the position of N circular ROIs 
		on a frame by clicking the center of each ROI
	"""

	# Callback function
	def add_roi_to_list(event, x, y, flags, data):
		if event == cv2.EVENT_LBUTTONDOWN:
			cv2.circle(data[0], (x, y), data[1], (0, 255, 0), 2) 
			cv2.circle(data[0], (x, y), data[1], (0, 0, 255), 3)  
			return data[2].append((x, y))

	ROIs = []

	# Start opencv window
	cv2.startWindowThread()
	cv2.namedWindow('detection')
	cv2.moveWindow("detection",100,100)   
	cv2.imshow('detection', frame)

	# create functions to react to clicked points
	data = [frame, radius, ROIs]
	cv2.setMouseCallback('detection', add_roi_to_list, data)  

	while len(ROIs) < n_rois:
		cv2.imshow('detection', frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
				break

	cv2.destroyAllWindows() 

	return ROIs

def split_channels(traces):
	"""
		Takes a dataframe with the signal for each fiber and splits the green, blue and violet frames. 
	"""
	columns = traces.columns

	new_data = {}
	for col in columns:

		blue = traces[col].values[0::3]
		green = traces[col].values[1::3]
		violet = traces[col].values[2::3]

		new_data[str(col)+"_blue"] = blue
		new_data[str(col)+"_green"] = green
		new_data[str(col)+"_isosbestic"] = violet

	new_data = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in new_data.items() ])) # video can cut off without having same amount of frames per color
	# # so replace those frames with Nans

	return pd.DataFrame(new_data)

################################################
#			    Signal processing    		   #
################################################

def smooth_edges(df, bin=10):
	df[:bin] = np.mean(df)
	df[-bin:] = np.mean(df)
	return df

def realign_columns(raw_traces, first_frame = 'blue'):
	'''
	Use when there are irregularities in signal identity; usually spotted when the first
	frame of the .avi does not show illumination in the  blue channel or the second frame
	does not show the blue channel dark; this function specifically realigns the traces dataframe 
	to begin with the blue channel, which is the way the rest of the code interprets the df
	'''

	if first_frame == 'green':
		cols = list(raw_traces.columns)
		ls = [x for x in zip(cols[2::3], cols[0::3], cols[1::3])]
		df = raw_traces[np.array(ls).ravel()]
		df.columns = df.columns.str.replace('isosbestic','temp', regex=True)
		df.columns = df.columns.str.replace('green','isosbestic', regex=True)
		df.columns = df.columns.str.replace('blue','green', regex=True)
		df.columns = df.columns.str.replace('temp','blue', regex=True)
		
		# roll over skipped values from skipped starting frames in df by 1
		
		# df.loc[max(df.index)+1, :] = None
		# for col in df.columns:
		# 	if 'blue' in col:
		# 		df[col] = df[col].shift(periods=1)

	if first_frame == 'isosbestic':
		cols = list(raw_traces.columns)
		ls = [x for x in zip(cols[1::3], cols[2::3], cols[0::3])]
		df = raw_traces[np.array(ls).ravel()]
		df.columns = df.columns.str.replace('green','temp', regex=True)
		df.columns = df.columns.str.replace('isosbestic','green', regex=True)
		df.columns = df.columns.str.replace('blue','isosbestic', regex=True)
		df.columns = df.columns.str.replace('temp','blue', regex=True)
	
		# same as above but for different starting frame

		# df.loc[max(df.index)+1, :] = None
		# for col in df.columns:
		# 	if 'blue' in col:
		# 		df[col] = df[col].shift(periods=1)
		# 	if 'green' in col: 
		# 		df[col] = df[col].shift(periods=1)
				
	return df






