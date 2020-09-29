import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import cv2
import os
import random
import sys
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt 
from pathlib import Path
import multiprocessing as mp
import tables
import scipy.io as sio
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import euclidean
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle #changed

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from utils.graphic_utils import *
from utils.detectron_utils import *
from utils.dataframe_utils import *

segmentation = 'panoptic'
hpc = False
PANOPTIC_DIR = '/mnt/raid/data/SCIoI/tim/panoptic-seg/'
RESULTS_DIR = '/mnt/raid/data/SCIoI/tim/'
DETECTRON2_DIR = '/mnt/antares_raid/home/tschroeder/detectron2/'
sys.path.append(DETECTRON2_DIR)
from demo.predictor import VisualizationDemo

def setup_cfg():
    cfg = get_cfg()
    if segmentation == 'panoptic':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    elif segmentation == 'mrcnn':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    if not hpc: 
        cfg.MODEL.DEVICE = 'cpu' #only on local device

    cfg.freeze()
    return cfg

def matchScore(x,y,mask):
    center = center_of_mass(mask)
    distanceFromCenter = euclidean(center,(y,x))
    size = np.count_nonzero(mask)
    maxDist = 1468
    maxSize = 921600
    score = (((1/(maxSize**2))*(size-maxSize)**2) + (1-distanceFromCenter/maxDist))/2
    return score

if __name__ == '__main__':

    cfg = setup_cfg() #Define Classifier for Panoptic Segmentation
    predictor = DefaultPredictor(cfg)
    demo = VisualizationDemo(cfg)
    metadata = demo.metadata

    GAZE_FOLDER_DIR = '/mnt/raid/data/SCIoI/GazeCom_ground_truth/'
    gaze_folder_list = [video for video in glob.glob(GAZE_FOLDER_DIR + '/*')]
    gaze_folder_list.sort()


    VIDEO_DIR = '/mnt/raid/data/SCIoI/GazeCom_videos/'
    video_path_list = [video for video in glob.glob(VIDEO_DIR + '/*')]
    video_path_list.sort()

    PROTO_OBJECTS = '/mnt/raid/data/SCIoI/tim/salData/'

    detect_path_list = [video for video in glob.glob(RESULTS_DIR + 'detect_video/' + '*.avi')]
    detect_path_list.sort()

    data_path_list = [video for video in glob.glob(PANOPTIC_DIR + '*_data.npy')]
    data_path_list.sort()

    info_path_list = [video for video in glob.glob(PANOPTIC_DIR + '*_info.npy')]
    info_path_list.sort()
    
    translate_dict = {'UNKNOWN':0, 'FIX':1, 'SACCADE':2, 'SP':3, 'NOISE':4}

    radius = 26.7 # 1 deg of visual angle is equvialent to radius of 13.35 pixel , 2 deg = 26.7 pixel ...
    fps=30

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    videos_run = [1]#[s for s in range(0,8)]


    markersize = 50
    maxframes = 600
    nframes = 595

    for v in tqdm(videos_run):#tqdm(range(videos_run), desc="videos"):
        for radius in [13.35]:#[13.35]:
            videoloc = detect_path_list[v] #+1  panoptic Segmented Video
            #vid = imageio.get_reader(videoloc,  'ffmpeg')
            vidlist = []
            

            video_name = os.path.splitext(os.path.basename(videoloc))[0]
            print(video_name)

            motion_proto_path = PROTO_OBJECTS + video_name + '.mat'
            motion_proto = sio.loadmat(motion_proto_path)
            motion_proto_maps = np.asarray(motion_proto['S'])

            gaze_dir = gaze_folder_list[v] #+1
            gaze_path_list = [video for video in glob.glob(gaze_dir + '/*')]
            gaze_path_list.sort()
            print(gaze_dir)
            print(data_path_list[v])
            print(info_path_list[v])

            # load panoptic segmentation results
            panoptic_seg = np.load(data_path_list[v], allow_pickle=True) # change storage.py line 134 for this
            segments_info = np.load(info_path_list[v],allow_pickle=True) ## change back later!! to v
            
            subjects_run = [s for s in range(25, len(gaze_path_list))]
            for s in tqdm(subjects_run, desc="subjects"):
                subject_name = os.path.splitext(os.path.basename(gaze_path_list[s]))[0][0:3]
                num_frames = min(maxframes,nframes, len(panoptic_seg))
                if not os.path.isdir(RESULTS_DIR +'df/' + video_name):
                    os.mkdir(RESULTS_DIR +'df/' + video_name)

                DF_NAME = os.path.join(RESULTS_DIR,'df/',video_name, video_name + '_' + subject_name + '_' + str(num_frames) + '_r='+str(radius)+'_mostSalient.pkl')
            
                gaze_path = gaze_path_list[s]
                print(gaze_path)
            # video_name = os.path.splitext(path_leaf(gaze_path_list[s]))[0]
                df = load_df(gaze_path)
                
                # go to either the full video or only through a certain number of frames
                df['class_id'] = ''
                df['thing'] = ''
                df['panoptic_score'] = np.nan
                df['motion_proto_id'] = ''
                df['proto_score'] = np.nan
                
                #df['walther_proto_id'] = ''

                
                for f in range(num_frames):
                    dtemp = df[(df['time'] >= 1e3*f/fps) & (df['time'] < 1e3*(f+1)/fps)] # assign data to frames
                    
                    for i in dtemp.index:
                        
                        # PANOPTIC 
                        x_coord = int(dtemp['x'][i])
                        y_coord = int(dtemp['y'][i])
                        track_id = -1
                        panoptic_score = 0
                        panoptic_mask = panoptic_seg[f].numpy()
                        segment_id = panoptic_mask[y_coord,x_coord]
                        isthing = segments_info[f][segment_id-1]['isthing'] #-1 because of indexing
                        category_id = segments_info[f][segment_id-1]['category_id']

                        y,x = np.ogrid[-y_coord:720-y_coord, -x_coord:1280-x_coord]
                        r=radius
                        mask = x*x + y*y <= r*r
                        
                        if 'track_id' in segments_info[f][segment_id-1]:
                            track_id = segments_info[f][segment_id-1]['track_id']

                        if isthing: #if point of interest is thing, label with thing_classes_label
                            label = metadata.thing_classes[category_id]
                            panoptic_score = matchScore(x_coord,y_coord,(panoptic_mask == segment_id))

                        else: #else check if stuff area of size (2*radius)^2 contains any other stuff or objects
                            area_of_interest = panoptic_seg[f].numpy()[mask] #circular mask to simulate foveate vision
                            aoi_segment_ids = np.unique(area_of_interest)
                            if len(aoi_segment_ids)>1: #if there are any other stuff or objects, check if the next most dominant one is stuff or thing
                                if segments_info[f][aoi_segment_ids[1]-1]['isthing']: # if the next most dominant one is thing, obtain with thing_class_label
                                    category_id = segments_info[f][aoi_segment_ids[1]-1]['category_id']
                                    label = metadata.thing_classes[category_id]
                                    isthing = True
                                    panoptic_score = matchScore(x_coord,y_coord,(panoptic_mask == aoi_segment_ids[1]))
                                else: # obtain with stuff_class_label
                                    label = metadata.stuff_classes[category_id]
                            else:
                                label = metadata.stuff_classes[category_id]

                        df.at[i,'class_id'] = label
                        df.at[i, 'thing'] = isthing
                        df.at[i, 'track_id'] = track_id
                        df.at[i, 'panoptic_score'] = panoptic_score
                        
                        dtemp.at[i, 'class_id'] = label
                        dtemp.at[i, 'thing'] = isthing
                        dtemp.at[i, 'track_id'] = track_id
                        dtemp.at[i, 'panoptic_score'] = panoptic_score

                        # MOTION PROTO OBJECT
                        proto_score = 0
                        motion_proto_id = 0
                        for n in range(0,5):
                            resized = cv2.resize(motion_proto_maps[f,:,:,n], (1280,720), interpolation = cv2.INTER_AREA)
                            blur = cv2.blur(resized,(20,20))
                            if blur[y_coord][x_coord] != 0:
                                motion_proto_id = n+1
                                proto_score = matchScore(x_coord,y_coord,blur)
                                break

                            area_of_interest = blur[mask] #same mask as for panoptic
                            aoi_segment_ids = np.unique(area_of_interest)
                            if len(aoi_segment_ids)>1:
                                motion_proto_id = n+1
                                proto_score = matchScore(x_coord,y_coord,blur)
                                break
                        
                        # Check most salient objects in panoptic segmentation
                        for segment in segments_info[f]:
                            if not segment['isthing']:
                                first_stuff_id = (segment['id'])
                                break
                            
                        labeled_thing_map = np.where(panoptic_mask<first_stuff_id,panoptic_mask,0) #All things to segment_id, stuff is 0
                        num_sal_objects = 5
                        mean_thing_saliency = []
                        mean_thing_saliency.append(0) #for 0 segment, as it is excluded in for loop 
                        for things_id in range(1,first_stuff_id-1): # exclude segment 0
                            single_thing_saliency = np.where(labeled_thing_map==things_id,1,0)*blur
                            mean_thing_saliency.append(np.mean(single_thing_saliency))
                        mean_thing_saliency = np.asarray(mean_thing_saliency)

                        max_sal_things = mean_thing_saliency.argsort()[-num_sal_objects:][::-1]
                        if segment_id in max_sal_things:
                            df.at[i,'most_salient_object'] = True
                        else:
                            df.at[i,'most_salient_object'] = False

                        """
                            # Check most salient objects in panoptic segmentation
                            center = center_of_mass(blur)
                            segment_id = panoptic_mask[int(center[0]),int(center[1])]
                            isthing = segments_info[f][segment_id-1]['isthing']
                            if isthing:
                                df.at[i,'most_salient_object'] = n+1
                            else:
                                df.at[i,'most_salient_object'] = 0
                        """
                        df.at[i,'motion_proto_id'] = motion_proto_id
                        df.at[i,'proto_score'] = proto_score
                        dtemp.at[i, 'motion_proto_id'] = motion_proto_id
                        dtemp.at[i, 'proto_score'] = proto_score
                        
                        """
                        walther_proto_id = 0
                        for n in range(0,4):
                            resized = cv2.resize(walther_proto_maps[f,:,:,n], (1280,720), interpolation = cv2.INTER_AREA)
                            blur = cv2.blur(resized,(20,20))
                            if blur[y_coord][x_coord] != 0:
                                walther_proto_id = n+1
                                break
                                
                            area_of_interest = blur[y_coord-radius:y_coord+radius,x_coord-radius:x_coord+radius]
                            aoi_segment_ids = np.unique(area_of_interest)
                            if len(aoi_segment_ids)>1:
                                walther_proto_id = n+1
                                break

                        df.at[i,'walther_proto_id'] = walther_proto_id
                        dtemp.at[i, 'walther_proto_id'] = walther_proto_id
                        """
                df.to_pickle(DF_NAME)   
                print('done!')
