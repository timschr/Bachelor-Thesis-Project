import pandas as pd
import numpy as np
import math

def load_df(gaze_path):
    """
    Load Gaze Data from arff file to pd Dataframe
    """
    
    from scipy.io.arff import loadarff
    arff = loadarff(gaze_path)
    df = pd.DataFrame.from_dict(arff[0])
    
    video_res_x = 1280
    video_res_y = 720
    df['time']-=df['time'][0]
    df['time']=df['time']/1000
    df['x'] = np.round(df['x'])
    df['y'] = np.round(df['y'])
    df = df[df['y'] >= 0]
    df = df[df['y'] < video_res_y]
    df = df[df['x'] >= 0]
    df = df[df['x'] < video_res_x]

    vid = gaze_path.split('/')[-2]
    subj = gaze_path.split('/')[-1][:3]
    df['video'] = vid
    df['subject'] = subj

    df = df.drop(columns=['handlabeller1', 'handlabeller2', 'confidence'])
    df = df.replace({'handlabeller_final': {0: 'UNKNOWN', 1.0:'FIX', 2.0:'SACCADE', 3.0:'SP', 4.0:'NOISE'}})
    return df

def merge_related_dp(df_func, label):
    """
    Merging all datpoints belonging to the label e.g. FIX. Returns pandas.Dataframe with properties of all for example fixation.
    
    df_func - Dataframe
    label - handlabeller_final Label e.g.: FIX 
    """
    d_return = df_func[df_func['handlabeller_final']==label] #from one [label] connect all datapoints
    counter = 0
    label_id = label + '_id'
    d_return[label_id] = np.nan
    d_return[label_id][d_return.index] = 0
    for i in range(1,len(d_return)):
        if d_return.index[i-1] == d_return.index[i]-1:
            d_return.at[d_return.index[i],label_id] = d_return[label_id][d_return.index[i]-1]
        else:
            counter += 1
            d_return[label_id][d_return.index[i]] = counter
    
    return d_return

# Segment Saccades
def make_df_saccade(df_func):
    d_sacc = merge_related_dp(df_func, 'SACCADE')
    d_fix = merge_related_dp(df_func, 'FIX')
    d_sp = merge_related_dp(df_func, 'SP')
    sacc_num_list = []
    sacc_num_list = d_sacc['SACCADE_id'].unique()
    df_saccade = pd.DataFrame()#sacc_num_list, columns = ['sacc_num'])


    for n in sacc_num_list:
        dtemp = d_sacc[d_sacc['SACCADE_id'] == n]
        i_start = dtemp.index.min()
        i_end = dtemp.index.max()
        x_delta = dtemp.at[i_end, 'x'] - dtemp.at[i_start, 'x']
        y_delta = dtemp.at[i_end, 'y'] - dtemp.at[i_start, 'y']
        df_saccade.at[n,'start_time'] = dtemp['time'][i_start]
        df_saccade.at[n,'amplitude'] = math.hypot(x_delta,y_delta)
        df_saccade.at[n,'duration'] = dtemp['time'].max()-dtemp['time'].min()
        df_saccade.at[n,'start_obj'] = dtemp['class_id'][i_start]
        df_saccade.at[n,'start_obj_id'] = dtemp['instance_id'][i_start]
        df_saccade.at[n,'start_is_thing'] = dtemp['thing'][i_start]
        df_saccade.at[n,'end_obj'] = dtemp['class_id'][i_end]
       # df_saccade.at[n,'end_obj_id'] = dtemp['instance_id'][i_end]
        df_saccade.at[n,'end_is_thing'] = dtemp['thing'][i_end]
        if i_end < len(df_func)-2: 
            if df_func.at[i_end+1, 'handlabeller_final'] == 'FIX':
                fix_id = d_fix.at[i_end+1, 'FIX_id']
                duration = len(d_fix[d_fix['FIX_id'] == fix_id])*4 #1 datapoint is 4ms
                df_saccade.at[n,'follow_by'] = 'FIX'
                df_saccade.at[n,'follow_duration'] = duration
            elif df_func.at[i_end+1, 'handlabeller_final'] == 'SP':
                sp_id = d_sp.at[i_end+1, 'SP_id']
                duration = len(d_sp[d_sp['SP_id'] == sp_id])*4
                df_saccade.at[n,'follow_by'] = 'SP'
                df_saccade.at[n,'follow_fix_duration'] = duration
            elif df_func.at[i_end+1, 'handlabeller_final'] == 'NOISE':
                df_saccade.at[n,'follow_by'] = 'NOISE'
                df_saccade.at[n,'follow_fix_duration'] = np.nan


        if dtemp['thing'][i_start]:
            if dtemp['thing'][i_end]:
                df_saccade.at[n,'which_to_which'] = 'T2T'
            else: 
                df_saccade.at[n,'which_to_which'] = 'T2S'
        else:
            if dtemp['thing'][i_end]:
                df_saccade.at[n,'which_to_which'] = 'S2T'
            else: 
                df_saccade.at[n,'which_to_which'] = 'S2S'
     
    df_saccade['same_object'] = ""
    df_saccade.loc[df_saccade['start_obj_id'] == df_saccade['end_obj_id'],'same_object'] = True
    df_saccade.loc[df_saccade['start_obj_id'] != df_saccade['end_obj_id'],'same_object'] = False
   

    df_saccade['velocity'] = df_saccade['amplitude'] /df_saccade['duration'] 
    return df_saccade