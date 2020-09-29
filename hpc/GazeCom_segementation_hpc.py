import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import os
import random
import sys
import glob
import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
import tables

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

DATASET = 'LUEBECK/'
hpc = True
segmentation = 'panoptic' # mrcnn or panoptic

DIR = Path(os.path.dirname(os.path.realpath('__file__'))) #.parent
print(DIR)
if (DIR== Path('/beegfs/home/users/t/tim.schroeder/')):
    CODE_DIR = os.path.join(DIR, 'object_rep/project_code/')
    hpc = True
else:
    CODE_DIR = DIR

print(CODE_DIR)
PROJECT_DIR = Path(CODE_DIR).parent
print(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'project_data/', DATASET)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'project_results/',DATASET+segmentation+'/')
if os.path.exists(RESULTS_DIR) == False:
    os.mkdir(RESULTS_DIR)

ROOT_DIR = os.path.join(CODE_DIR, "detectron2/")

sys.path.append(ROOT_DIR)
from demo.predictor import VisualizationDemo
print('HPC:', hpc)

def setup_cfg():
    cfg = get_cfg()
    if segmentation == 'panoptic':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = ROOT_DIR + "/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_model.pkl"
    elif segmentation == 'mrcnn':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = '/home/users/t/tim.schroeder/object_rep/project_code/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_model.pkl'
    if not hpc:
        cfg.MODEL.DEVICE = 'cpu' #only on local device

    cfg.freeze()
    return cfg


if __name__ == "__main__":
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    demo = VisualizationDemo(cfg)

    video_path_list = [video for video in glob.glob(DATA_DIR + '/*')]
    video_path_list.sort()
    for video_path in [video_path_list[12]]:
        output_data = []
        output_info = []
        
        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = basename = os.path.splitext(os.path.basename(video_path))[0]
        
        if os.path.isdir(RESULTS_DIR):
            output_fname = os.path.join(RESULTS_DIR, basename)
            output_fname = os.path.splitext(output_fname)[0] + ".avi"
            output_data_fname = os.path.join(RESULTS_DIR, basename)
            output_data_fname = os.path.splitext(output_data_fname)[0] + "_data.npy"
            output_info_fname = os.path.join(RESULTS_DIR, basename)
            output_info_fname = os.path.splitext(output_info_fname)[0] + "_info.npy"

        else:
            output_fname = RESULTS_DIR

        #assert not os.path.isfile(output_fname), output_fname
        
        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*'XVID'),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
        
        #assert os.path.isfile(video_path)
        if segmentation == 'panoptic':
            for vis_frame,panoptic_seg, segments_info in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                output_file.write(vis_frame)
                output_data.append(panoptic_seg)
                output_info.append(segments_info)

        elif segmentation == 'mrcnn':
            #create .h5py outfile
            filename = RESULTS_DIR+basename+'_out.h5'
            f = tables.open_file(filename, mode='w')
            atom = tables.Float64Atom()
            f.create_earray(f.root, 'pred_boxes', atom, (0, 4))
            f.create_earray(f.root, 'scores', atom, (0, 1))
            f.close()
            
            for vis_frame, pred in tqdm.tqdm(demo.run_on_video(video)):
                #predictions = [pred_boxes, pred_classes, pred_masks, scores]
                output_file.write(vis_frame)
                #output_data.append(pred)
                a = pred[0].tensor.numpy()
                b = pred[1].numpy()
                #reshape results
             
                boxes = np.empty((100,4))
                boxes[:] = np.nan
                boxes[:a.shape[0],:a.shape[1]] = a
                scores = np.empty((100))
                scores[:] = np.nan
                scores[:b.shape[0]] = b

                f = tables.open_file(filename, mode='a')
                f.root.pred_boxes.append(boxes)
                f.root.scores.append(scores.reshape((100, 1)))
            
               
                f.close()

        video.release()
        output_file.release()

        if segmentation == 'panoptic':
            np.save(output_info_fname, output_info, allow_pickle=True)
            np.save(output_data_fname, output_data, allow_pickle=True)
            del output_data
            del output_info

