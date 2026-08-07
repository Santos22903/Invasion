import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plotGFs(xy_df):
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
    axes[0].set_title("X coord")
    axes[1].set_title("Y coord")

    handles0 = []
    handles1 = []
    labels   = []
    for step in steps:
        labelText = "Step " + str(int(step))
        data2plot = xy_df.xs(key=step,drop_level=True)
        #
        sns.kdeplot(x=data2plot.x.to_numpy(),ax=axes[0], label=labelText)
        sns.kdeplot(x=data2plot.y.to_numpy(),ax=axes[1], label=labelText)

    axes[0].legend(loc=1,fontsize='small')
    axes[1].legend(loc=1,fontsize='small')
    axes[1].axes.get_yaxis().get_label().set_visible(False) # removing y-axis label from right plot
    plt.savefig(outFile)

def checkDiffusion(gfCoords,x0,y0,max_diff_time):
    """
    01/28/22  Routine for checking diffusion
    gfCoords - DF of GF coords,
    x0,y0    - "origin", wrt which discplacement happens
    max_diff_time - time to which to run stepping
    diff_delt - diffusion step time
    """
    global diff_delt
    diff_time = 0
    step = 0
    steps2plot = [3, 10, 20, 50, 100, 1000]
    plotColors = ["red","blue","green","black","silver","violet","cyan"]

    fig, axes = plt.subplots(1,3,figsize=(30,10))
    axes[0].set_title("X coord")
    axes[1].set_title("Y coord")
    axes[2].set_title("Absolute distance")
    max_diff_time = 30
    while diff_time < max_diff_time:
        step+=1
        #print("step ",step)
        diff_time += diff_delt
        gfCoords = RWstep(gfCoords)
        # plotting part
        if (step in steps2plot):
            print("step: ",step)
            print("steps2plot ",steps2plot)
            stepIndex = steps2plot.index(step)
            print(stepIndex)
            print(type(stepIndex))
            labelText = str(round(diff_time,2)) + "seconds"
            sns.distplot(ax=axes[0],a=gfCoords.x, hist=False, color=plotColors[stepIndex], label=labelText,
                         bins=np.linspace(start=-100,stop=100,num=102))#,kde_kws={'cut':0})
            sns.distplot(ax=axes[1],a=gfCoords.y, hist=False, color=plotColors[stepIndex], label=labelText,
                         bins=np.linspace(start=-100,stop=100,num=102))#,kde_kws={'cut':0})
            # displacement wrt "origin"
            delta_x = np.array(gfCoords.x) - x0; delta_y = np.array(gfCoords.y) - y0
            tmp1 = np.square(delta_x)+np.square(delta_y)
            gfCoordsR = np.sqrt(tmp1.astype(float))
            print(gfCoordsR)
            sns.distplot(ax=axes[2],a=gfCoordsR, hist=False, color=plotColors[stepIndex], label=labelText)#,
            #             bins=np.linspace(start=0,stop=100,num=51))#,kde_kws={'cut':0})

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.savefig("diffusion_coords_randseed50_N1000.pdf")    
