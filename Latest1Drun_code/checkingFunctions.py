import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

def plotGFs_1d(x_df):
    """
    plotting x of diffusing growth factors - separate line for different steps
    input: data frame with columns "x" - "y" - "step#"
    """
    plt.rc('ytick',  labelsize=6)
    plt.rc('axes',   labelsize=8)
    plt.rc('figure', titlesize=12)
    
    outFile = "gf_coords_"+time.strftime("%Y%m%d")+".pdf"
    x_df.set_index(keys='step',inplace=True)
    steps = list(set(x_df.index.values))
    print("steps: ",steps)
    print("AFTER")
    steps.sort()
    #print(type(steps))
    #print(type(steps[1]))    
    fig, axes = plt.subplots(1,1)#,figsize=(70,50))
    #fig.tight_layout()
    axes.set_title('X')

    handles0 = []
    handles1 = []
    labels   = []
    for step in steps:
        labelText = "Step " + str(int(step))
        data2plot = x_df.xs(key=step,drop_level=True)
        #
        print("data2plot ",data2plot)
        print("data2plot.x ",data2plot.x)
        print("type(data2plot.x) ",type(data2plot.x))
        sns.kdeplot(x=data2plot.x,ax=axes, label=labelText)
        #sns.kdeplot(x=data2plot.y,ax=axes[1], label=labelText)
    print("AFTER LOOP")
    axes.legend(loc=1,fontsize='small')
    #axes[1].legend(loc=1,fontsize='small')
    #axes[1].axes.get_yaxis().get_label().set_visible(False) # removing y-axis label from right plot
    plt.savefig(outFile)

def plotGFs_3d(xyz_df):
    """
    Input: stepping df in 3d (x/y/z/step)
    Drops on of coordinate columns and calls plotGFs for resulting df
    """
    print("xyz_df ",xyz_df)
    xy_df = xyz_df.drop('z',axis=1)
    print("xy_df ",xy_df)
    plotGFs(xy_df,'X','Y')

def plotGFs(xy_df,coord1_name,coord2_name):
    """
    plotting x&y of diffusing growth factors - separate line for different steps
    input: data frame with columns "x" - "y" - "step#"
    """
    plt.rc('ytick',  labelsize=6)
    plt.rc('axes',   labelsize=8)
    plt.rc('figure', titlesize=12)
    
    outFile = "gf_coords_"+time.strftime("%Y%m%d")+".pdf"
    xy_df.set_index(keys='step',inplace=True)
    steps = list(set(xy_df.index.values))
    steps.sort()
    #print(type(steps))
    #print(type(steps[1]))    
    fig, axes = plt.subplots(1,2)#,figsize=(70,50))
    #fig.tight_layout()
    axes[0].set_title(coord1_name)
    axes[1].set_title(coord2_name)

    handles0 = []
    handles1 = []
    labels   = []
    for step in steps:
        labelText = "Step " + str(int(step))
        data2plot = xy_df.xs(key=step,drop_level=True)
        #
        print("data2plot ",data2plot)
        print(data2plot.x)
        print(type(data2plot.x))
        sns.kdeplot(x=data2plot.x,ax=axes[0], label=labelText)
        sns.kdeplot(x=data2plot.y,ax=axes[1], label=labelText)

    axes[0].legend(loc=1,fontsize='small')
    axes[1].legend(loc=1,fontsize='small')
    axes[1].axes.get_yaxis().get_label().set_visible(False) # removing y-axis label from right plot
    plt.savefig(outFile)


