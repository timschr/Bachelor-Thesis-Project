import numpy as np
import matplotlib.pyplot as plt 
"""
Making graphics to analyse data
"""

def make_saccade_histgramms(df_func, path):
    """
    Graphic divided into four parts: 
        1. Which percentage of saccades goes from Thing to Stuff (T2S),T2T,S2S,S2T
        2. Which percentage starts/ends on thing vs. stuff
        3. Count of all Class IDs on start/end points of saccades
        4. Mean velocity of saccades subdevided in T2T,S2T etc.

    df_func - dataframe including:
        - coordinates: x,y
        - class_id: Class Label of specific coordinate
        - thing: Boolean is true if class_id belongs to category things

    path - Graphic save path
    """
    
    id_count = (df_func['start_obj'].value_counts() + df_func['end_obj'].value_counts())
    object_count = df_func['start_is_thing'].value_counts() + df_func['end_is_thing'].value_counts()
    which_to_which_count = df_func['which_to_which'].value_counts()

    mode_list = ['T2T','T2S','S2T','S2S']
    velocity_list = []
    for s in mode_list:
        mean_velocity = df_func[df_func['which_to_which']==s]['velocity'].mean()
        velocity_list.append(mean_velocity)
        
    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax1, ax2, ax3, ax4 = axes.ravel()
    
    fig.set_size_inches(15, 10)
    
    ax3.bar(id_count.index, id_count.values)
    ax3.set_title('class id in start/end objects')

    ax1.pie(which_to_which_count.values, labels=which_to_which_count.index, autopct='%1.1f%%')
    ax1.set_title('saccade from/to stuff vs. thing')

    ax4.bar(mode_list, velocity_list)
    ax4.set_title('mean velocity')
    
    
    ax2.pie(object_count.values, autopct='%1.1f%%')
    ax2.set_title('saccade start/end point is thing vs. stuff')
    ax2.legend(ax2,labels=object_count.index, bbox_to_anchor=(1,0.9), loc="top right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
                

    plt.setp( ax3.xaxis.get_majorticklabels(), rotation=60 )
    
    plt.tight_layout()
    plt.savefig(path, dpi = 300)
    
    
def make_histgramms(df, path):
    """
    Graphic divided into four parts: 
        1. Which percentage of gaze allocation is on things vs stuff
        2. On which objects are these things divided?
        3. Count of all Class IDs 
        4. Density Map of all coordinates

    df - dataframe including:
        - coordinates: x,y
        - class_id: Class Label of specific coordinate
        - thing: Boolean is true if class_id belongs to category things

    path - Graphic save path
    """

    id_count = df['class_id'].value_counts()
    is_thing_count = df['thing'].value_counts()
    object_count = df[df['thing'] == True]['class_id'].value_counts()
    
    id_count = id_count[id_count.values>len(df)*0.01]
    #object_count = object_count[object_count.values>len(df)*0.01]

    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax1, ax2, ax3, ax4 = axes.ravel()
    
    fig.set_size_inches(15, 10)
    
    ax3.bar(id_count.index, id_count.values)
    ax3.set_title('class id count')

    ax1.pie(is_thing_count.values, labels=is_thing_count.index, autopct='%1.1f%%')
    ax1.set_title('is_thing')

    ax4 = sns.kdeplot(df.x, 720-df.y, shade=True,shade_lowest=False, cmap="Reds",bw=.2)
    ax4.set_title('points of interest')
    
    
    ax2.pie(object_count.values, autopct='%1.1f%%')
    ax2.set_title('things')
    ax2.legend(ax2,labels=object_count.index, bbox_to_anchor=(1,0.9), loc="top right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
                

    plt.setp( ax3.xaxis.get_majorticklabels(), rotation=60 )
    
    plt.tight_layout()
    plt.savefig(path, dpi = 300)
    
    #plt.show()

def nested_pie_chart_panoptic(df_short, path):
    plt.clf()
    df_short = df_short[df_short['handlabeller_final'] != 'NOISE']
    df_short = df_short[df_short['handlabeller_final'] != 'SACCADE']
    is_thing_count = df_short['thing'].value_counts()
    thing_count = df_short[df_short['thing'] == True]['class_id'].value_counts()
    stuff_count = df_short[df_short['thing'] == False]['class_id'].value_counts()

    # Make data: I have 2 groups and N subgroups
    if (len(is_thing_count) == 2):
        group_names=['thing', 'stuff']
        group_size=[is_thing_count[True],is_thing_count[False]]
    else:
        group_names=['stuff']
        group_size=[is_thing_count[False]]
    if ('thing' in group_names):
        higher_thres_thing = thing_count[thing_count>is_thing_count[True]*0.05]
        lower_thres_thing = thing_count[thing_count<is_thing_count[True]*0.05]

    higher_thres_stuff = stuff_count[stuff_count>is_thing_count[False]*0.1]
    lower_thres_stuff = stuff_count[stuff_count<is_thing_count[False]*0.1]

    if len(thing_count>1):
        subgroup_names = [*(thing_count[:len(higher_thres_thing)].index),'others',*(stuff_count[:len(higher_thres_stuff)].index),'others']
        subgroup_size = [*(thing_count[:len(higher_thres_thing)].values),np.sum(lower_thres_thing.values), *(stuff_count[:len(higher_thres_stuff)].values),np.sum(lower_thres_stuff.values)]
    else:
        subgroup_names = [*(thing_count.index),'others',*(stuff_count[:len(higher_thres_stuff)].index),'others']
        subgroup_size = [*(thing_count.values), *(stuff_count[:len(higher_thres_stuff)].values),np.sum(lower_thres_stuff.values)]
    
    # Create colors
    a, b =[plt.cm.Blues, plt.cm.Reds]
    if ('thing' in group_names):
        color_list_thing = [a(0.6-0.1*x) for x in range(len(higher_thres_thing)+1)]
    else:
        color_list_thing = []

    color_list_stuff = [b(0.6-0.1*x) for x in range(len(higher_thres_stuff)+1)]

    bigger = plt.pie(group_size, labels=group_names, colors=[a(0.8), b(0.8)],
                    startangle=90, frame=True, autopct='%1.1f%%', pctdistance = 0.85)
    smaller = plt.pie(subgroup_size, labels=subgroup_names,
                    colors=[*color_list_thing,*color_list_stuff], radius=0.7,
                    startangle=90, labeldistance=0.58, rotatelabels = True, textprops = dict(size = 'x-small'))
    centre_circle = plt.Circle((0, 0), 0.4, color='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
        
    plt.axis('equal')
    
    # save it
    plt.tight_layout()
    plt.savefig(path, dpi = 200)

def nested_pie_chart_salient_panoptic(df_short, path):
   
    df_short = df_short[df_short['handlabeller_final'] != 'NOISE']
    df_short = df_short[df_short['handlabeller_final'] != 'SACCADE']
    is_thing_count = df_short['thing'].value_counts()
    thing_count = df_short[df_short['thing'] == True]['most_salient_object'].value_counts()
    stuff_count = df_short[df_short['thing'] == False]['most_salient_object'].value_counts()

    # Make data: I have 2 groups and N subgroups
    if (len(is_thing_count) == 2):
        group_names=['thing', 'stuff']
        group_size=[is_thing_count[True],is_thing_count[False]]
    else:
        group_names=['stuff']
        group_size=[is_thing_count[False]]
    if (len(thing_count) != 0):
        higher_thres_thing = thing_count[thing_count>is_thing_count[True]*0.05]
        lower_thres_thing = thing_count[thing_count<is_thing_count[True]*0.05]

    #higher_thres_stuff = stuff_count[stuff_count>is_thing_count[False]*0.1]
    #lower_thres_stuff = stuff_count[stuff_count<is_thing_count[False]*0.1]

    
    subgroup_names = [*(thing_count.index),*(stuff_count.index)]
    subgroup_size = [*(thing_count.values), *(stuff_count.values)]
   
    # Create colors
    a, b =[plt.cm.Blues, plt.cm.Reds]
    
    color_list_thing = [a(0.6-0.1*x) for x in range(len(thing_count))]
    color_list_stuff = [b(0.6-0.1*x) for x in range(len(stuff_count))]

    bigger = plt.pie(group_size, labels=group_names, colors=[a(0.8), b(0.8)],
                    startangle=90, frame=True, autopct='%1.1f%%', pctdistance = 0.85)
    smaller = plt.pie(subgroup_size, labels=subgroup_names,
                    colors=[*color_list_thing,*color_list_stuff], radius=0.7,
                    startangle=90, labeldistance=0.58, rotatelabels = True, textprops = dict(size = 'x-small'))
    centre_circle = plt.Circle((0, 0), 0.4, color='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
        
    plt.axis('equal')
    
    # save it
    plt.tight_layout()
    plt.savefig(path, dpi = 200)



def nested_pie_chart_proto(df_short, path):
    plt.clf()
    df_short = df_short[df_short['handlabeller_final'] != 'NOISE']
    df_short = df_short[df_short['handlabeller_final'] != 'SACCADE']

    proto_id_count = df_short['motion_proto_id'].value_counts()[df_short['motion_proto_id'].value_counts().index != 0]
    background_count = df_short['motion_proto_id'].value_counts()[df_short['motion_proto_id'].value_counts().index == 0]


    group_names=['proto object', 'background']
    group_size=[sum(proto_id_count.values),sum(background_count.values)]
    proto_id_count = df_short['motion_proto_id'].value_counts()[df_short['motion_proto_id'].value_counts().index != 0]
    background_count = df_short['motion_proto_id'].value_counts()[df_short['motion_proto_id'].value_counts().index == 0]

    subgroup_names = [*(proto_id_count.index),'']
    subgroup_size = [*(proto_id_count.values), *(background_count.values)]

    # Create colors
    a, b =[plt.cm.Blues, plt.cm.Reds]

    color_list_thing = [a(0.6-0.1*x) for x in range(len(proto_id_count))]
    color_list_stuff = [b(0.8)]

    bigger = plt.pie(group_size, labels=group_names, colors=[a(0.8), b(0.8)],
                    startangle=90, frame=True, autopct='%1.1f%%', pctdistance = 0.85)
    smaller = plt.pie(subgroup_size, labels=subgroup_names,
                    colors=[*color_list_thing,*color_list_stuff], radius=0.7,
                    startangle=90, labeldistance=0.7)
    centre_circle = plt.Circle((0, 0), 0.4, color='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
        
    plt.axis('equal')
    
    # save it
    plt.tight_layout()
    plt.savefig(path, dpi = 200)

def nested_pie_chart_walther_proto(df_short, path):
    plt.clf()
    df_short = df_short[df_short['handlabeller_final'] != 'NOISE']
    df_short = df_short[df_short['handlabeller_final'] != 'SACCADE']

    proto_id_count = df_short['walther_proto_id'].value_counts()[df_short['walther_proto_id'].value_counts().index != 0]
    background_count = df_short['walther_proto_id'].value_counts()[df_short['walther_proto_id'].value_counts().index == 0]


    group_names=['proto object', 'background']
    group_size=[sum(proto_id_count.values),sum(background_count.values)]
    proto_id_count = df_short['walther_proto_id'].value_counts()[df_short['walther_proto_id'].value_counts().index != 0]
    background_count = df_short['walther_proto_id'].value_counts()[df_short['walther_proto_id'].value_counts().index == 0]

    subgroup_names = [*(proto_id_count.index.astype(int)),'']
    subgroup_size = [*(proto_id_count.values), *(background_count.values)]

    # Create colors
    a, b =[plt.cm.Blues, plt.cm.Reds]

    color_list_thing = [a(0.6-0.1*x) for x in range(len(proto_id_count))]
    color_list_stuff = [b(0.8)]

    bigger = plt.pie(group_size, labels=group_names, colors=[a(0.8), b(0.8)],
                    startangle=90, frame=True, autopct='%1.1f%%', pctdistance = 0.85)
    smaller = plt.pie(subgroup_size, labels=subgroup_names,
                    colors=[*color_list_thing,*color_list_stuff], radius=0.7,
                    startangle=90, labeldistance=0.7)
    centre_circle = plt.Circle((0, 0), 0.4, color='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
        
    plt.axis('equal')
    
    # save it
    plt.tight_layout()
    plt.savefig(path, dpi = 200)
    

def nested_pie_chart_total(df2, path):
    plt.clf()
    df2 = df2[df2['handlabeller_final'] != 'NOISE']
    df2 = df2[df2['handlabeller_final'] != 'SACCADE']
    post_saccade_state = df2['post_saccade_state'].value_counts().sort_index(ascending=True)

    group_names=[*post_saccade_state.index[(post_saccade_state.index != '') & (post_saccade_state.index != 4)]]
    group_size=[*post_saccade_state.values[(post_saccade_state.index != '') & (post_saccade_state.index != 4)]]

    states = df2['post_saccade_state'].value_counts().sort_index(ascending=True).index#[1:5]
    subgroup_names = []
    subgroup_size = []
    for state in states:
        subgroup_names.append(df2[df2['post_saccade_state'] == state]['handlabeller_final'].value_counts().sort_index(ascending=True).index)
        subgroup_size.append(df2[df2['post_saccade_state'] == state]['handlabeller_final'].value_counts().sort_index(ascending=True).values)
    subgroup_names = np.concatenate(subgroup_names)
    subgroup_names[subgroup_names == 'SACCADE'] = ['SAC']
    #subgroup_names[subgroup_names == 'NOISE'] = ['*']


    # Create colors
    a,b,c,d = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges]
    colors = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges]
    color_list = []
    for i in range(len(subgroup_size)):
        color_list.append([colors[i](0.6-0.1*x) for x in range(len(subgroup_size[i]))])

    """
    color_list_1 = [a(0.6-0.1*x) for x in range(len(subgroup_size[0]))]
    color_list_2 = [b(0.6-0.1*x) for x in range(len(subgroup_size[1]))]
    color_list_3 = [c(0.6-0.1*x) for x in range(len(subgroup_size[2]))]
    color_list_4 = [d(0.6-0.1*x) for x in range(len(subgroup_size[3]))]
    """
    color_list = np.concatenate(color_list)
    subgroup_size = np.concatenate(subgroup_size)

    bigger = plt.pie(group_size, labels=group_names, 
                    startangle=90, frame=True, colors=[a(0.8),b(0.8),c(0.8),d(0.8)], autopct='%1.1f%%', pctdistance = 0.85)
    smaller = plt.pie(subgroup_size, labels=subgroup_names,
                        colors=[*color_list], radius=0.7,
                        startangle=90, labeldistance=0.7, rotatelabels = True, textprops = dict(size = 'x-small')) #_1,*color_list_2,*color_list_3,*color_list_4
    centre_circle = plt.Circle((0, 0), 0.4, color='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
        
    plt.axis('equal')
    plt.axis('off')
    # save it
    plt.tight_layout()
    plt.savefig(path,dpi = 200)
