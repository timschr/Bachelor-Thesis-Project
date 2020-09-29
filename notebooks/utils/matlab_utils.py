import scipy.io as sio
import numpy as np

def load_proto_objects(PROTO_OBJECT_DIR, video_name):

    motionData_path = PROTO_OBJECT_DIR + 'myData/' + video_name + '.mat'
    waltherData_path = PROTO_OBJECT_DIR + 'waltherData/' + video_name + '.mat'

    motionData_raw = sio.loadmat(motionData_path)
    waltherData_raw = sio.loadmat(waltherData_path)

    motionProto = np.asarray(motionData_raw['S'])                               #.shape = (28, 50, 596)
    waltherProto = np.asarray(waltherData_raw['S'])                             #.shape = (28, 50, 596)

    return waltherProto, motionProto

def load_saliency_maps(PROTO_OBJECT_DIR, video_name):
    motionSaliency_path = PROTO_OBJECT_DIR + 'mySaliency/' + video_name + '.mat'
    waltherSaliency_path = PROTO_OBJECT_DIR + 'waltherSaliency/' + video_name + '.mat'

    motionSaliency_raw = sio.loadmat(motionSaliency_path)
    waltherSaliency_raw = sio.loadmat(waltherSaliency_path)

    motionSaliency = np.asarray(motionSaliency_raw['S'])                               #.shape = (28, 50, 596)
    waltherSaliency = np.asarray(waltherSaliency_raw['S'])                             #.shape = (28, 50, 596)

    return waltherSaliency, motionSaliency