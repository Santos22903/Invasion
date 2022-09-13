#!/usr/bin/env python3
"""
Author: Svyatoslav Tkachenko
Date:   12/07/21

Simulation of cells producing a public benefit
growth factor invading a population of 
non-producers

"""
import sys
import importlib
import numpy as np
from scipy import constants
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
import multiprocessing as mp

cores = mp.cpu_count() - 1
seed = 50
rng = np.random.default_rng(seed)  # can be called without a seed

DIFFUSION_DEBUG = False

def initializeGlobals():
    """
    Parameters of the simulation used by different functions: physical/lattice
    constants, time scales
    """
    ################## Constants read in from config file
    global P_mut,Rprod0,inflx,steep,n_cycles,L,L_filled,secrete_tstep,secreteN
    global diff_coef, delx, k_on, k_off, receptors_per_cell, um_to_dm

    if len(sys.argv)>2:
        sys.exit("Usage: %s [.py config-file basename]" % sys.argv[0])
    elif len(sys.argv)==2:
        conf_file = sys.argv[1]
    else:
        conf_file = "default_cfg"
    print("Configuration file used: %s" % conf_file)
    try:
        modules = importlib.import_module(conf_file)
        P_mut         = modules.P_mut;         inflx    = modules.inflx
        steep         = modules.steep;         n_cycles = modules.n_cycles;
        L             = modules.L;             L_filled =modules.L_filled;
        secrete_tstep = modules.secrete_tstep; secreteN = modules.secreteN
        diff_coef     = modules.diff_coef;     delx     = modules.delx
        k_on          = modules.k_on;          k_off    = modules.k_off
        receptors_per_cell = modules.receptors_per_cell
        um_to_dm           = modules.um_to_dm
    except ImportError:
        raise ImportError('Cannot import the config file')
    ### Checking input
    assert L>=L_filled, "Filled part can't be larger than lattice"
    ###
    ################## Requiring calculation using read-in constants
    global diff_coef_scaled, diff_delt, k_on_norm

    ### diffusion parameters
    diff_coef_scaled = diff_coef/pow(delx,2) # to use with dx=1 on lattice, delx expected in um => 1/s unit
    diff_delt = 1/(2*2*diff_coef_scaled)     # diffusion time step, s (see, e.g., "lecture6"); 2nd "2" is dimension of lattice
    #
    node_volume = pow((delx*um_to_dm),3) # if not mistaken, need "reaction volume" for k_on scaling; converted to dm (k_on has dm^3)
    k_on_norm = k_on/(constants.Avogadro*node_volume) # normalized for Gillespie
    
    return

def iniSquareLattice():    
    """
    Initializing square lattice of size LxL: empty nodes (if L>L_filled)
    are 0, non-producers are 1, the seeded producer is 2.
    L_filled x L_filled in the middle filled with non-producers
    Center of filled - single producer
    """
    global L,L_filled
    #
    if L==L_filled:
        iniLattice = np.ones((L,L))
    else:
        iniLattice = np.zeros((L,L))
        loFilled = (L-L_filled)//2
        hiFilled = loFilled + L_filled
        iniLattice[loFilled:hiFilled,loFilled:hiFilled] = 1
    iniLattice[(L-1)//2,(L-1)//2] = 2 # since filled part always in the middle
    #iniLattice[0,0]=2 # for testing
    if DIFFUSION_DEBUG:
        print(iniLattice)
    return iniLattice

def iniBenefitArray():
    """
    Initializing array of "bound benefits" - now much benefit (e.g. molecules of
    growth factors) is bound in each node - the final state of this array will 
    be used for "updating strategies" at the proliferation time
    """
    global L
    benefArray = np.zeros((L,L))
    if DIFFUSION_DEBUG:
        print(benefArray)
    return(benefArray)
    
def RWstep(gfcoords):
    """
    perform random walk step: accept a list of positions,
    "step", return new positions
    """
    global L # lattice size
    #print("Old gfcoords:\n",gfcoords)
    # generate df of "Steps" - +/-1 df of same dimension as gf coords
    possSteps = np.arange(-1,2,2)
    nGFs      = len(gfcoords.index)
    #stepDF = pd.DataFrame({'x': np.random.choice(possSteps,size=nGFs),
    #                       'y': np.random.choice(possSteps,size=nGFs)})
    stepDF = pd.DataFrame({'x': rng.choice(possSteps,size=nGFs),
                           'y': rng.choice(possSteps,size=nGFs)})
    stepDF.index = gfcoords.index # otherwise NaNs
    #print("stepDF:\n",stepDF)
    # get new position by summing old+step
    gfcoords = gfcoords + stepDF
    #print("New gfcoords:\n",gfcoords)
    # weeding out of boundary gf's (absorbing b.c.)
    gfcoords = gfcoords[(gfcoords['x']>=0) & (gfcoords['y']>=0) &
                        (gfcoords['x']<L) & (gfcoords['y']<L)]
    #print("Filtered gfcoords:\n",gfcoords)

    return(gfcoords)

def bindingStep_mp(benefarray, gfcoords):
    cores = mp.cpu_count() - 1
    print("Using ",cores," cores")
    #
    coordmat = coord_to_mat(gfcoords.to_numpy(dtype="int"))
    benefarray_1d = benefarray.flatten(); coordmat_1d = coordmat.flatten()

    pool = mp.Pool(cores)

    input = zip(benefarray_1d, coordmat_1d)
    #print(list(input))
    resarray = np.array(list(pool.starmap(GillespieFn,input)),dtype="int")
    print("resarray")
    print(resarray)
    
    pool.close()
def bindingStep(benefarray, gfcoords, useMultiproc=True, useMap=True):
    """
    Routine performing binding after each diffusion time step. Goes over the whole
    lattice. Flattening -> list comprehension way
    ? generating for the whole lattice simultaneously to speed up ?
    
    Input: benefarray - how much "public good" already bound at each node
           gfcoords   - coordinates of free floating "goodies"
           useMap     - defines if list comprehension (False) or mapping
                        (True) is used for running GillespieFn()
    Output: 1st 2 inputs after possible binding in the diff_delt time
            (time it takes to travel unit step - b/n nodes)
    """

    global L # lattice size
    
    # 04/21/22 here call the routine running binding procedure for each node
    # expect "+1" if new association at the node, "-1" if new dissociation
    # at the node, "0" if nothing changes
    # ? makes it possibly to run in lapply style later and return a DF ?
    # ? with possible parallel processing ?

    # 04/21/22 learning multiprocessing for later, just call for 1 node now
    # converting to "lattice matrix"
    coordmat = coord_to_mat(gfcoords.to_numpy(dtype="int"))

    #print("Before gillespie: benefarray[3,3] ",benefarray[3,3]," coordmat[3,3] ",coordmat[3,3])
    #boundben, freeben = GillespieFn(benefBound=benefarray[3,3],benefFree=coordmat[3,3])
    #print("After gillespie: boundben ",boundben," freeben ",freeben)

    # way 1: flatten -> list comrehension -> fromiter -> reshape back    
    benefarray_1d = benefarray.flatten(); coordmat_1d = coordmat.flatten()
    if useMultiproc:
        print("multiprocessing")
        pool = mp.Pool(cores)
        resarray = np.array(pool.starmap(GillespieFn,zip(benefarray_1d,coordmat_1d)),dtype="int")
        pool.close()
    elif useMap:
        resarray = np.array(list(map(GillespieFn,benefarray_1d, coordmat_1d)),dtype="int")
    else:
        resarray = np.array([GillespieFn(x,y) for x,y in zip(benefarray_1d,coordmat_1d)],dtype="int")
    
    benefarray_1d = resarray[:,0]; benefarray = benefarray_1d.reshape(L,L)
    coordmat_1d = resarray[:,1];   coordmat   = coordmat_1d.reshape(L,L)

    # converting back to dataframe
    gfcoords = mat_to_coord(coordmat)

    return benefarray, gfcoords
    
    
def GillespieFn(benefBound,benefFree):
    """
    Incorporating Gillespie routine, k_on normalized a la Gabhann/Popel
    Input: benefBound - number of "goodies" already bound (b4 this step) by node/cell,
                        can dissociate, benefarray or boundBenefitArray elsewhere
           benefFree  - number of "goodies" that just diffused here, can bind
                        gfcoords or coordmat elsewhere
    Output: 1) benefBound, 2) benefFree after the step
    """
    global k_on_norm, k_off, diff_delt, receptors_per_cell
    
    #print("benefBound ",benefBound," benefFree ",benefFree)

    # if no bound or free PGs, nothing to do
    if(benefBound==0 and benefFree==0):
        return 0, 0
    
    # step numbers in comments below correspond to Gillespie, 1977 paper
    # 0) initialization
    curr_step_time = 0 # time from the beginning of the step

    ### looping over steps 1-3 until next time beyond diffusion time step
    while 1:
        # 1) calculating "a's" (see paper), a1 - association, a2 - dissociation
        a1 = benefFree*receptors_per_cell*k_on_norm
        a2 = k_off * benefBound
        a0 = a1 + a2

        # 2) generating random numbers, finding time/reaction
        rands = rng.random.random_sample(size=2)
        #print("rands")
        #print(rands)
        #rands[0] = 0.98
        tau = (1/a0)*np.log(1/rands[0])
        curr_step_time = curr_step_time + tau        
        if(curr_step_time>diff_delt):
            return (benefBound,benefFree)
        if rands[1]*a0 <= a1:
            print("\nASSOCIATION\n")
            print("rands[0] ",rands[0]," rands[1] ",rands[1])
            print("a1 ",a1," a2 ",a2," tau ",tau) 
            benefBound = benefBound + 1
            benefFree  = benefFree  - 1
        else:
            benefBound = benefBound - 1
            benefFree  = benefFree  + 1
    ###
    #return (benef_bound,benef_free)
    
    
    
    

def doStepping(benefarray,
               gfcoords
               ):
    """
    Routine performing GF stepping: combined diffusion by random walk (RW)
    and possible binding at each RW step.
    Stepping performed until 1) all bind OR 2) all diffuse past lattice OR
    3) time for next secretion came.
    """
    # needed globals
    global secrete_tstep, diff_delt
    
    # satisfying the last condition by calculating max step number as the number
    # of diffusion steps fitting in the secretion delt
    # secrete_delt expected in s, in config file
    # diff_delt should be seconds, see comment to calculating it and above
    Nsteps_max = round(secrete_tstep/diff_delt)
    Nsteps_max=1
    if DIFFUSION_DEBUG:
        print("Nsteps_max ",Nsteps_max)
        xy_df_to_plot = pd.DataFrame({'x':[],'y':[],'step':[]},dtype=int)
        steps2plot = [10,20,50,100,Nsteps_max-1]

    # loop over steps checking conditions 1&2 after each step
    sumbenefs = 0
    #print("Nsteps_max ",Nsteps_max)
    for stepN in range(Nsteps_max):
        #print("step ",stepN)
        ## GF's drifting here
        gfcoords = RWstep(gfcoords)
        if DIFFUSION_DEBUG and (stepN in steps2plot): # preparing to plot FG coords for some steps if debugging
            stepdata = pd.DataFrame({'x':gfcoords.x,'y':gfcoords.y, 'step':[stepN]*len(gfcoords.x)})            
            xy_df_to_plot = xy_df_to_plot.append(stepdata,ignore_index=True)
            
        ## check if any GF's left on lattice (condition 2)
        if(len(gfcoords)==0):            
            print("triggered")
            return(benefarray)
        
        ## binding (or not, to cell to which drifted)
        print(stepN)
        (benefarray, gfcoords) = bindingStep(benefarray, gfcoords, useMultiproc=False, useMap=False)
        #(benefarray, gfcoords) = bindingStep_mp(benefarray, gfcoords)

        #print("benefarray")
        #print(benefarray)
        #print(np.sum(benefarray))        
        
        if(np.sum(benefarray)>sumbenefs):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("step ",stepN)
            print("np.sum(benefarray) ",np.sum(benefarray))
            print("sumbenefs ",sumbenefs)
            sumbenefs = np.sum(benefarray)
            print("benefarray after Gillespie:")
            print(benefarray)
            print("coordmat after Gillespie:")
            print(coord_to_mat(gfcoords.to_numpy(dtype="int")))
        
    if DIFFUSION_DEBUG: # plotting prepared GF coords if debugging
        xy_df_to_plot = xy_df_to_plot.apply(pd.to_numeric, errors='coerce')# get rid of "object" types
        xy_df_to_plot = xy_df_to_plot.astype("int",errors="ignore")        # step was still "float"
        print(xy_df_to_plot)
        plotGFs(xy_df_to_plot)

    return benefarray

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
        
def secreteGF(cells,oldGfDf):
    """
    "secreting" growth factors by producers: where a producer ("2") is 
    present, its coordinates are assigned to added secreteN GF's
    """
    global secreteN
    
    producerCoords = np.where(cells == 2)
    xcol = np.repeat(producerCoords[0],secreteN)
    ycol = np.repeat(producerCoords[1],secreteN)
    newGfData = {'x': xcol,
                 'y': ycol}
    newGfDf = pd.DataFrame(newGfData)
    # if not ignore_index, row indices will be repeated after concat
    totalGfDf = pd.concat([oldGfDf,newGfDf],ignore_index=True)
    """
    print("producerCoords ",producerCoords)
    print("type(oldGfDf) ",type(oldGfDf))
    print("oldGfDf")
    print(oldGfDf)
    print("type(newGfDf) ",type(newGfDf))
    print("newGfDf")
    print(newGfDf)
    print("totalGfDf")
    print(totalGfDf)
    """

    return(totalGfDf)

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

def coord_to_mat(coord_array):
    global L
    zero_arr = np.zeros((L,L),dtype=int)
    np.add.at(zero_arr, tuple(coord_array.T), 1)
    return zero_arr

def mat_to_coord(lattice_mat):
    lattice_mat = lattice_mat.astype(int)
    idx = np.transpose(np.nonzero(lattice_mat))
    coord_out = np.repeat(idx,lattice_mat[tuple(idx.T)],axis=0)
    coordDF = pd.DataFrame(data=coord_out, columns = ['x','y'])
    return coordDF
    
def main():
    # initialize vars from config file
    global P_mut,Rprod0,inflx,steep,n_cycles,L,L_filled,secrete_tstep,secreteN
    global diff_coef, delx, k_on, k_off, receptors_per_cell
    # global vars not from config file
    global diff_coef_scaled, diff_delt, k_on_norm
    initializeGlobals()
    print("Constants from the configuration file")
    print("P_mut,inflx,steep,n_cycles,L,L_filled in main",P_mut,inflx,steep,n_cycles,L,L_filled)
    print("diff_coef, delx, k_on, k_off in main ", round(diff_coef,3), delx, k_on, k_off)
    print("receptors_per_cell in main ",receptors_per_cell) 
    print("Calculated global constants")
    print("diff_coef_scaled, diff_delt, k_on_norm in main ",
          round(diff_coef_scaled,3), round(diff_delt,3),f'{k_on_norm:.3}')

    # random seed for debugging, remove later
    #np.random.seed(50)
    #print(np.random.get_state())
    
    # initialize/setup lattice of producers/non-producers/empty nodes
    cellArray = iniSquareLattice()

    # initialize bound benefit lattice (how much benefit is bound in each node)
    # at the moment I do not see how it can be different from all 0's, but won't hurt
    boundBenefitArray = iniBenefitArray()

    # initialize growth factor secretion; since initialization,
    # old GF dataframe is empty
    gfCoords = secreteGF(cellArray,oldGfDf=pd.DataFrame(columns=['x','y']))

    # convert gfCoords to lattice matrix
    #coordmat = coord_to_mat(gfCoords.to_numpy(dtype="int32"))
    # convert lattice matrix to coordinate matrix
    #gfCoords = mat_to_coord(coordmat)
    
    # diffusion
    #checkDiffusion(gfCoords=gfCoords,x0=gfCoords.iloc[0,0],
    #               y0=gfCoords.iloc[0,1],max_diff_time = 200,diff_delt=diff_delt)

    # Stepping over diffusion-binding until all bind OR all diffuse past lattice OR
    # time for next secretion came
    start = timer()
    boundBenefitArray = doStepping(benefarray=boundBenefitArray,
                                   gfcoords=gfCoords
                                   )
    end = timer()
    print("Time of doStepping: ",end-start)
    

if __name__=="__main__":
    main()
