#!/usr/bin/env python3
"""
Author: Svyatoslav Tkachenko
Date:   12/07/21

Simulation of cells producing a public benefit
growth factor invading a population of 
non-producers

04/28/22 Changed Gillespie routine to accept random number pair as well.
         (not needed if not multiprocessing, but to be uniformly consistent,
         besides, generating a bunch of rnds right away should not make it slower)

07/19/22 Backed up before starting adding run pver cycles (in parallel right away)

07/27/22 Correction to Gillespie - if free or bound absent, only the pther reacton goes

08/16/22 Adding option of runs in parallel or not, mainly for debugging at this point,
         although possible to use in normal running
08/24/22 To accelerate, read from config file secrete_one_tstep - time step of secreting
         each molecule. secreteN will mean how many molecules to secrete at once, so
         time step of secretion will be a product of these
04/20/23 Introduce swapping of binding and RWstep in stepping so that GFs have a chance
         to bind to producer efore diffusing away. The difference with previous versions
         w step0 would be keeping the condition on quitting stepping loop if no free GFs
         present
09/29/23 making 1d version
"""
import sys
import argparse
import importlib
import numpy as np
from scipy import constants
import pandas as pd
import time
from timeit import default_timer as timer
import multiprocessing as mp
import functools
import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import date

from utilFunctions     import coord_to_mat1d, mat_to_coord1d, getFactor
#from checkingFunctions import plotGFs_1d
from checkDiffusion    import plotDiffusion1d
from benefitFns        import hillFn
from updating          import wrightFisherSpatial1D

import platform

print("VERSION IS ",platform.python_version())

cores = mp.cpu_count() - 1
rngseed = None                          # seed for debugging
rng = np.random.default_rng()  # can be called without a seed

rng_proc1 = []
rng_proc2 = []
rng_proc3 = []
rng_proc4 = []
rng_proc5 = []
## number of bound and unbound GF before each proliferation step
bound_sums_list   = []
unbound_sums_list   = []
## fractions of bound by producers/non-producers
prod_bound_frac_list = []
nonprod_bound_frac_list = []
## number of cell cycles after which invaded
invasion_cycleN = []
perish_cycleN = []
## to determine how to scale total bound sum to size increase, fill histo
boundSumList = []
## spatial profile data frame
df_spatial_free_gf = pd.DataFrame({"count":[]})
df_spatial_bound_gf= pd.DataFrame({"count":[]})
## max/min benefits
max_benef_list = []
min_benef_list = []

### Altered 09/14/23 to accomodate 3D
def getSpatialProfile(gfcoords_df, initial_spatial):
    global x0
    #print("gfcoords \n",gfcoords_df)
    df_x_distances = pd.DataFrame({"x_dist":   abs(gfcoords_df["x"]-x0)})
    
    #print("df_xy_distances \n",df_xy_distances)
    df_distances = pd.DataFrame({"distance": df_x_distances["x_dist"]})
    #print("df_distances \n",df_distances)
    df_temp_profile = df_distances.groupby(["distance"]).agg(count=pd.NamedAgg(column="distance",aggfunc="count"))
    #print("df_temp_profile \n",df_temp_profile)
    ## got profile befire "this" proliferation; merge it with overall profile tracing DF
    ## to make a histogram after all proliferations
    #print("initial_spatial before \n",initial_spatial)
    final_spatial = initial_spatial.add(df_temp_profile,fill_value=0)
    #print("final_spatial after\n",final_spatial)

    return(final_spatial)
    
def initializeGlobals(logfile_arg,D_arg,kon_arg,koff_arg,cost_arg,conf_file):
    """
    Parameters of the simulation used by different functions: physical/lattice
    constants, time scales
    09/14/23 fixing calculated vars
    """
    ################## Constants read in from config file
    global n_cycles,L,L_filled,cellcycle_t
    global secrete_one_tstep,secreteN,cost,useCost
    global delx, receptors_per_cell, um_to_dm
    global diff_coef, k_on, k_off
    global EC50_number_per_node, n_hill, EC50_bound_in_cycle
    global bind_1reaction, useMultiprocConstant, useMapConstant
    global invasion_done_perc, n_runs, cores2use
    global runs_in_parallel, benef_frac_out, cycleN_out, boundSum_out, GF_sums_list
    global benefFactor, gf_spat_profile, benef_size_out
    global bound_PG_offset, bound_benef_offset, use_bound_benef_offset
    global fixedSize, expandDet
    global out2screen, logfile

    global DIFFUSION_DEBUG, BINDING_DEBUG, RNG_DEBUG, CC_DEBUG

    diff_coef = D_arg; k_on = kon_arg; k_off = koff_arg; cost = cost_arg; logfile = logfile_arg

    #print("Process: ", mp.current_process())
    #print("initglobs")
    #print(f"conf_file is {conf_file}\n\n\n\n")
    
    #if len(sys.argv)>2:
    #    sys.exit("Usage: %s [.py config-file basename]" % sys.argv[0])
    #elif len(sys.argv)==2:
    #    conf_file = sys.argv[1]
    #else:
    #    conf_file = "def_cfg_gen"

    try:
        modules                = importlib.import_module(conf_file)
        n_cycles               = modules.n_cycles;
        L                      = modules.L
        L_filled               = modules.L_filled;
        cellcycle_t            = modules.cellcycle_t
        secrete_one_tstep      = modules.secrete_one_tstep
        secreteN               = modules.secreteN
        #cost                   = modules.cost
        useCost                = modules.useCost
        #diff_coef              = modules.diff_coef
        delx                   = modules.delx
        #k_on                   = modules.k_on
        #k_off                  = modules.k_off
        receptors_per_cell     = modules.receptors_per_cell
        um_to_dm               = modules.um_to_dm
        EC50                   = modules.EC50
        n_hill                 = modules.n_hill
        bind_1reaction         = modules.bind_1reaction # set to False for good
        useMultiprocConstant   = modules.useMultiproc   # ?could be useful for GillespieFn, worth checking for low D? then cores->cores2use
        useMapConstant         = modules.useMap         # set to True for good
        invasion_done_perc     = modules.invasion_done_perc
        n_runs                 = modules.n_runs
        ######### For parallel running; considering what python does, probably obsolete
        cores2use              = modules.cores2use        # could be utilized for gillespie in ||; check this??
        runs_in_parallel       = modules.runs_in_parallel
        #########
        benef_frac_out         = modules.benef_frac_out
        cycleN_out             = modules.cycleN_out
        boundSum_out           = modules.boundSum_out
        GF_sums_list           = modules.GF_sums_list
        gf_spat_profile        = modules.gf_spat_profile
        benef_size_out         = modules.benef_size_out
        bound_PG_offset        = modules.bound_PG_offset
        bound_benef_offset     = modules.bound_benef_offset
        use_bound_benef_offset = modules.use_bound_benef_offset
        fixedSize              = modules.fixedSize
        expandDet              = modules.expandDet
        out2screen             = modules.out2screen
        DIFFUSION_DEBUG        = modules.DIFFUSION_DEBUG
        BINDING_DEBUG          = modules.BINDING_DEBUG
        RNG_DEBUG              = modules.RNG_DEBUG
        CC_DEBUG               = modules.CC_DEBUG
    except ImportError:
        raise ImportError('Cannot import the config file')
    ### Checking input
    assert L>=L_filled, "Filled part can't be larger than lattice"
    ###
    ################## Requiring calculation using read-in constants
    global diff_coef_scaled, diff_delt, k_on_norm, secrete_tstep
    global benef_offset

    ### diffusion parameters
    diff_coef_scaled = diff_coef/pow(delx,2) # to use with dx=1 on lattice, delx expected in um => 1/s unit
    #diff_delt = 1/(2*2*diff_coef_scaled)     # diffusion time step, s (see, e.g., "lecture6"); 2nd "2" is dimension of lattice
    diff_delt = 1/(2*1*diff_coef)      # 07/14/22 nothing else is scaled, removing from here (see param write-up); changing second "2" (dimension) to "1" for 1D modeling
    #
    node_volume = pow((delx*um_to_dm),3) # if not mistaken, need "reaction volume" for k_on scaling; converted to dm (k_on has dm^3)
    k_on_norm = k_on/(constants.Avogadro*node_volume) # normalized for Gillespie
    #
    # converting EC50 to number of ligands "around" each node
    EC50_numb_per_liter = EC50 * constants.Avogadro # EC50 in Molar=moles/Liter
    EC50_number_per_node = round(EC50_numb_per_liter * node_volume) # node_volume in dm3=L
    EC50_bound_in_cycle = round(EC50_number_per_node * receptors_per_cell * k_on_norm * cellcycle_t) #reaction rate*time of cell cycle
    #
    # programmatic secretion time step - product of biologicalk secretion time step and # of molecules to secrete at once
    secrete_tstep = secrete_one_tstep*secreteN
    #
    # get factor to divide benefit sum for propotional expansion
    benefFactor = getFactor(k_on,diff_coef)
    #
    # map benefit offset from # of bound PGs to "benefits" to use it as "offset" argument later
    if use_bound_benef_offset:
        benef_offset = bound_benef_offset
    else:
        benef_offset = hillFn(n_bound=bound_PG_offset, EC50_bound=EC50_number_per_node, n=n_hill, offset=0)
    
    return

def iniLinearLattice():    
    """
    Initializing linear lattice of size L: empty nodes (if L>L_filled)
    are 0, non-producers are 1, the seeded producer is 2.
    L_filled in the middle filled with non-producers
    Center of filled - single producer
    """
    global L,L_filled
    global x0 # coordinates of "initial" "2" to use as origin wrt which spatial profile wil be calculated
    global logfile
    #
    if L==L_filled:
        iniLattice = np.ones((L))
    else:
        iniLattice = np.zeros((L))
        loFilled = (L-L_filled)//2
        hiFilled = loFilled + L_filled
        iniLattice[loFilled:hiFilled] = 1
    x0 = (L-1)//2
    iniLattice[x0] = 2 # since filled part always in the middle

    if DIFFUSION_DEBUG:
        if logfile=="__stdout__":
            logstream = sys.stdout
        else:
            logstream = open(logfile,'a')
        print(f"Initial producer coordinate is {x0}", file=logstream)
        print(iniLattice, file=logstream)
        if logstream is not sys.stdout:
            logstream.close()
    return iniLattice

def iniBenefitArray():
    """
    Initializing array of "bound benefits" - now much benefit (e.g. molecules of
    growth factors) is bound in each node - the final state of this array will 
    be used for "updating strategies" at the proliferation time
    """
    global L
    global logfile
    benefArray = np.zeros((L))
    if DIFFUSION_DEBUG:
        if logfile=="__stdout__":
            logstream = sys.stdout
        else:
            logstream = open(logfile,'a')
        print(benefArray, file=logstream)
        if logstream is not sys.stdout:
            logstream.close()
    return(benefArray)

def RWstep_v2(local_rng,gfcoords):
    """
    simplified/generalized version of stepping
    Stepping:
    1) prepare np array of zeros: #GF's by #dimensions
    2) generate 1d array of steps (+/- 1): length=#GF's
    3) generate array of dimension # in which step performed, same length
    4) Assign generated in (2) to places in (1) defined by (3), convert to PD df
       to add to input (initial) coordinate DF
    """
    ndim      = 1
    colnames  = ['x']
    #
    global L
    nGFs      = len(gfcoords.index)

    # Making stepping DF (steps 1-4 above)
    # 1)
    stepDF_np = np.zeros((nGFs,ndim))
    # 2)
    possSteps = np.arange(-1,2,2)
    step_vec  = local_rng.choice(possSteps,size=nGFs)
    # 3)
    possDims  = np.arange(ndim)
    dim_vec   = local_rng.choice(possDims,size=nGFs)
    # 4)
    stepDF_np[np.arange(nGFs),dim_vec] = step_vec
    stepDF_pd = pd.DataFrame(stepDF_np, columns = colnames)
    # adding, checking boundary conditions
    stepDF_pd.index = gfcoords.index # otherwise NaNs(tries to match indices when adding)
    gfcoords = gfcoords + stepDF_pd
    gfcoords = gfcoords[~(gfcoords>=L).any(axis=1)]
    gfcoords = gfcoords[~(gfcoords<0).any(axis=1)]

    return(gfcoords)

def bindingStep(local_rng, benefarray, gfcoords, cellarray, useMultiproc=True, useMap=True):
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

    global L              # lattice size
    global bind_1reaction # if True, call gillespie for 1 reaction and return from it
    
    # converting to "lattice matrix"
    coordmat = coord_to_mat1d(gfcoords.to_numpy(dtype="int"),L)

    benefarray_1d = benefarray; coordmat_1d = coordmat
    cellarray_1d = cellarray

    rng_list = [local_rng]*len(benefarray_1d)
    resarray = np.array(list(map(GillespieFn,rng_list,benefarray_1d,coordmat_1d,cellarray_1d)),dtype="int")

    benefarray_1d = resarray[:,0]; benefarray = benefarray_1d
    coordmat_1d = resarray[:,1];   coordmat   = coordmat_1d

    # converting back to dataframe
    gfcoords = mat_to_coord1d(coordmat)

    #print("after mat_to_coord")
    
    return benefarray, gfcoords

def GillespieFn_multi(benefBound,benefFree,cell,local_seed):
    """
    Separate routine for multiprocessing when random number generator seed/state
    needs to be passed - i.e., more than one reaction
    Incorporating Gillespie routine, k_on normalized a la Gabhann/Popel
    Input: benefBound - number of "goodies" already bound (b4 this step) by node/cell,
                        can dissociate, benefarray or boundBenefitArray elsewhere
           benefFree  - number of "goodies" that just diffused here, can bind
                        gfcoords or coordmat elsewhere
    Output: 1) benefBound, 2) benefFree after the step
    """
    global k_on_norm, k_off, diff_delt, receptors_per_cell
    global logfile
    
    # if node not filled with cell, nothing to do
    if cell==0:
        return 0, benefFree # can return benefBound,benefFree, nothing should be bound there
    # if no bound or free PGs, nothing to do
    if(benefBound==0 and benefFree==0):
        return 0, 0

    local_rng = np.random.default_rng(local_seed)
    
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
        rands = local_rng.random(size=2); rand0=rands[0]; rand1=rands[1]
        #print("rands")
        #print(rands)
        #rands[0] = 0.98
        tau = (1/a0)*np.log(1/rand0)
        curr_step_time = curr_step_time + tau        
        if(curr_step_time>diff_delt):
            return (benefBound,benefFree)
        #if rand1*a0 <= a1:
        ## 07/29 addtiom
        if (rand1*a0 <= a1 and benefFree>0) or (benefBound==0):
            if BINDING_DEBUG:
                if logfile=="__stdout__":
                    logstream = sys.stdout
                else:
                    logstream = open(logfile,'a')
                print("\nASSOCIATION\n", file=logstream)
                print("rand0 ",rand0," rand1 ",rand1, file=logstream)
                print("a1 ",a1," a2 ",a2," tau ",tau, file=logstream)
                if logstream is not sys.stdout:
                    logstream.close()
            benefBound = benefBound + 1
            benefFree  = benefFree  - 1
        else:
            benefBound = benefBound - 1
            benefFree  = benefFree  + 1

def GillespieFn_1react(benefBound,benefFree,cell,rand0,rand1):
    """
    Same as GillespieFn(), but only 1 reaction - done and exit
    """
    #print("benefBound ",benefBound," benefFree ",benefFree," cell ",cell)
    
    global k_on_norm, k_off, diff_delt, receptors_per_cell
    global logfile
    # if node not filled with cell, nothing to do
    if cell==0:
        return 0, benefFree # can return benefBound,benefFree, nothing should be bound there
    # if no bound or free PGs, nothing to do
    if(benefBound==0 and benefFree==0):
        return 0, 0
    ### step 0 omitted, 1 reaction only - not tracing time
    # 1) calculating "a's" (see paper), a1 - association, a2 - dissociation
    a1 = benefFree*receptors_per_cell*k_on_norm
    a2 = k_off * benefBound
    a0 = a1 + a2
    
    # 2) generating random numbers, finding time/reaction
    tau = (1/a0)*np.log(1/rand0)
    if(tau>diff_delt):
        return (benefBound,benefFree)
    #if rand1*a0 <= a1:
    ## 07/29 addition
    if (rand1*a0 <= a1 and benefFree>0) or (benefBound==0):
        benefBound = benefBound + 1
        benefFree  = benefFree  - 1
        if BINDING_DEBUG:
            if logfile=="__stdout__":
                logstream = sys.stdout
            else:
                logstream = open(logfile,'a')
            print("\nASSOCIATION\n", file=logstream)
            print("rand0 ",rand0," rand1 ",rand1, file=logstream)
            print("a1 ",a1," a2 ",a2," tau ",tau, file=logstream) 
            print("benefBound ",benefBound, file=logstream)
            print("benefFree ",benefFree, file=logstream)
            if logstream is not sys.stdout:
                logstream.close()
    else:
        benefBound = benefBound - 1
        benefFree  = benefFree  + 1
        
    return (benefBound,benefFree)
        
def GillespieFn(local_rng,benefBound,benefFree,cell):#,rand0,rand1):
    """
    Incorporating Gillespie routine, k_on normalized a la Gabhann/Popel
    Input: benefBound - number of "goodies" already bound (b4 this step) by node/cell,
                        can dissociate, benefarray or boundBenefitArray elsewhere
           benefFree  - number of "goodies" that just diffused here, can bind
                        gfcoords or coordmat elsewhere
    Output: 1) benefBound, 2) benefFree after the step
    """
    global k_on_norm, k_off, diff_delt, receptors_per_cell
    global logfile
    #print("local_rng ",local_rng," benefBound ",benefBound," benefFree ",benefFree," cell ",cell)

    # if node not filled with cell, nothing to do
    if cell==0:
        return 0, benefFree # can return benefBound,benefFree, nothing should be bound there
    
    # if no bound or free PGs, nothing to do
    if(benefBound==0 and benefFree==0):
        return 0, 0
    
    # step numbers in comments below correspond to Gillespie, 1977 paper
    # 0) initialization
    curr_step_time = 0 # time from the beginning of the step

    ### looping over steps 1-3 until next time beyond diffusion time step
    #print("before while")
    iter_ctr = 0
    while 1:
        if iter_ctr>5:
            return (benefBound,benefFree)
        # 1) calculating "a's" (see paper), a1 - association, a2 - dissociation
        a1 = benefFree*receptors_per_cell*k_on_norm
        a2 = k_off * benefBound
        a0 = a1 + a2

        # 2) generating random numbers, finding time/reaction
        #print("before generation")
        rands = local_rng.random(size=2)
        #print("rands: ",rands)
        rand0=rands[0]; rand1=rands[1]
        if RNG_DEBUG:
            #print("Filling lists for histos")
            #print("mp.current_process() ",mp.current_process().name)
            if mp.current_process().name=='SpawnPoolWorker-1':
                #print("In 1")
                rng_proc1.extend([rand0,rand1])
                #print("mp.current_process() ",mp.current_process()," rng_proc1 ",rng_proc1)
            elif mp.current_process().name=='SpawnPoolWorker-2':
                #print("In 2")                
                rng_proc2.extend([rand0,rand1])
                #print("mp.current_process() ",mp.current_process()," rng_proc1 ",rng_proc2)
            elif mp.current_process().name=='SpawnPoolWorker-3':
                #print("In 3")
                rng_proc3.extend([rand0,rand1])
                #print("mp.current_process() ",mp.current_process()," rng_proc1 ",rng_proc3)
            elif mp.current_process().name=='SpawnPoolWorker-4':
                #print("In 4")
                rng_proc4.extend([rand0,rand1])
                #print("mp.current_process() ",mp.current_process()," rng_proc1 ",rng_proc4)
            elif mp.current_process().name=='SpawnPoolWorker-5':
                #print("In 5")
                rng_proc5.extend([rand0,rand1])
                #print("mp.current_process() ",mp.current_process()," rng_proc1 ",rng_proc5)

        tau = (1/a0)*np.log(1/rand0)
        curr_step_time = curr_step_time + tau
        #print("tau ",tau," curr_step_time ",curr_step_time," diff_delt ",diff_delt," benefBound ",benefBound," benefFree ",benefFree)
        if(curr_step_time>diff_delt):
            #print("curr_step_time>diff_delt")
            return (benefBound,benefFree)
        ## 07/29 addition: if free or bound absent, only the other one happens
        if (rand1*a0 <= a1 and benefFree>0) or (benefBound==0):
            if BINDING_DEBUG:
                if logfile=="__stdout__":
                    logstream = sys.stdout
                else:
                    logstream = open(logfile,'a')
                print("\nASSOCIATION\n", file=logstream)
                print("rand0 ",rand0," rand1 ",rand1, file=logstream)
                print("a1 ",a1," a2 ",a2," tau ",tau, file=logstream)
                if logstream is not sys.stdout:
                    logstream.close()
            benefBound = benefBound + 1
            benefFree  = benefFree  - 1
        else:
            benefBound = benefBound - 1
            benefFree  = benefFree  + 1
        iter_ctr = iter_ctr + 1
    

def doStepping(local_rng,
               benefarray,
               gfcoords,
               cellarray
               ):
    """
    Routine performing GF stepping: combined diffusion by random walk (RW)
    and possible binding at each RW step.
    Stepping performed until 1) all bind OR 2) all diffuse past lattice OR
    3) time for next secretion came.
    """
    
    # needed globals
    global secrete_tstep, diff_delt, L
    global useMultiprocConstant, useMapConstant # using || running or map/list comprehension
    global logfile
    
    # satisfying the last condition by calculating max step number as the number
    # of diffusion steps fitting in the secretion delt
    # secrete_delt expected in s, in config file
    # diff_delt should be seconds, see comment to calculating it and above
    Nsteps_max = round(secrete_tstep/diff_delt)
    #Nsteps_max=1
    if DIFFUSION_DEBUG:
        if logfile=="__stdout__":
            logstream = sys.stdout
        else:
            logstream = open(logfile,'a')
        print("Nsteps_max ",Nsteps_max, file=logstream)
        if logstream is not sys.stdout:
            logstream.close()
        x_df_to_plot = pd.DataFrame({'x':[]},dtype=int)
        steps2plot = [10,20,50,100,Nsteps_max-1]

    # loop over steps checking conditions 1&2 after each step
    sumbenefs = 0
    #print("Nsteps_max ",Nsteps_max)
    for stepN in range(Nsteps_max):
        #if stepN%100 == 0:
        #    print("step ",stepN)
        #print(cellarray)
        #print(benefarray)
        #print(gfcoords)
        ## GF's drifting here
        #print("before rwstep\n",gfcoords)
            
        ## check if any GF's left on lattice (condition 2)
        if (len(gfcoords)==0 and np.sum(benefarray)==0):            
            #print("No GFs left on lattice condition triggered")
            if DIFFUSION_DEBUG:
                x_df_to_plot = x_df_to_plot.apply(pd.to_numeric, errors='coerce')# get rid of "object" types
                x_df_to_plot = x_df_to_plot.astype("int",errors="ignore")        # step was still "float"
                if logfile=="__stdout__":
                    logstream = sys.stdout
                else:
                    logstream = open(logfile,'a')
                print(x_df_to_plot, file=logstream)
                if logstream is not sys.stdout:
                    logstream.close()
                x_dict = x_df_to_plot.groupby('step')['x'].apply(list).to_dict()
                plotDiffusion1d(x_dict,sidelength=L)
                #sys.exit(0)
            return (benefarray,gfcoords)
        
        ## binding (or not, to cell to which drifted)
        #print(stepN)
        #print("before binding step\n",gfcoords)
        (benefarray, gfcoords) = bindingStep(local_rng,benefarray, gfcoords, cellarray,
                                             useMultiproc=useMultiprocConstant, useMap=useMapConstant)
        #print("after binding step\n",gfcoords)

        if len(gfcoords)>0:
            gfcoords = RWstep_v2(local_rng,gfcoords)
        #print("after rwstep\n",gfcoords)
        if DIFFUSION_DEBUG and (stepN in steps2plot): # preparing to plot FG coords for some steps if debugging
            stepdata = pd.DataFrame({'x':gfcoords.x, 'step':[stepN]*len(gfcoords.x)})            
            x_df_to_plot = x_df_to_plot.append(stepdata,ignore_index=True)

    if DIFFUSION_DEBUG: # plotting prepared GF coords if debugging
        x_df_to_plot = x_df_to_plot.apply(pd.to_numeric, errors='coerce')# get rid of "object" types
        x_df_to_plot = x_df_to_plot.astype("int",errors="ignore")        # step was still "float"
        if logfile=="__stdout__":
            logstream = sys.stdout
        else:
            logstream = open(logfile,'a')
        print(x_df_to_plot, file=logstream)
        if logstream is not sys.stdout:
            logstream.close()
        x_dict = x_df_to_plot.groupby('step')['x'].apply(list).to_dict()
        plotDiffusion1d(x_dict,sidelength=L)

    #print("before returning\n",gfcoords)
        
    return benefarray, gfcoords
        
def secreteGF(cells,oldGfDf):
    """
    "secreting" growth factors by producers: where a producer ("2") is 
    present, its coordinates are assigned to added secreteN GF's
    """
    global secreteN
    global logfile
    
    producerCoords = np.where(cells == 2)
    xcol = np.repeat(producerCoords[0],secreteN)
    newGfData = {'x': xcol}
    newGfDf = pd.DataFrame(newGfData)

    # if not ignore_index, row indices will be repeated after concat
    totalGfDf = pd.concat([oldGfDf,newGfDf],ignore_index=True)
    if DIFFUSION_DEBUG:
        if logfile=="__stdout__":
            logstream = sys.stdout
        else:
            logstream = open(logfile,'a')
        print("producerCoords ",producerCoords,flush=True,file=logstream)
        print("oldGfDf ",oldGfDf,flush=True,file=logstream)
        print("newGfDf ",newGfDf,flush=True,file=logstream)
        print("totalGfDf ",totalGfDf,flush=True,file=logstream)
        if logstream is not sys.stdout:
            logstream.close()

    return(totalGfDf)

def printGlobalVars():
    global logfile
    # global vars from config file
    global n_cycles,L,L_filled,cellcycle_t
    global secrete_tstep,secreteN,cost, useCost
    global diff_coef, delx, k_on, k_off, receptors_per_cell
    global EC50_number_per_node, n_hill, EC50_bound_in_cycle
    global bind_1reaction, useMultiprocConstant, useMapConstant
    global invasion_done_perc, n_runs, cores2use
    global bound_PG_offset, use_bound_benef_offset, bound_benef_offset, benef_offset
    global runs_in_parallel, benef_frac_out, cycleN_out, boundSum_out, boundDF_b4_prolif, unboundDF_b4_prolif
    global fixedSize, expandDet
    global out2screen, DIFFUSION_DEBUG, BINDING_DEBUG, RNG_DEBUG, CC_DEBUG
    # global vars not from config file
    global diff_coef_scaled, diff_delt, k_on_norm
    if logfile == "__stdout__":
        logstream = sys.stdout
    else:
        logstream = open(logfile,'a')
    print("\n",flush=True, file=logstream)
    print("Constants from the configuration file",flush=True, file=logstream)
    print("n_cycles,L,L_filled in main: ",n_cycles,L,L_filled,flush=True, file=logstream)
    print("cellcycle_t, secrete_tstep, secreteN: ", cellcycle_t, secrete_one_tstep, secreteN,flush=True, file=logstream)
    print("cost, useCost: ",cost, useCost, flush=True, file=logstream)
    print("diff_coef, delx, k_on, k_off in main ", round(diff_coef,3), delx, k_on, k_off,flush=True, file=logstream)
    print("receptors_per_cell in main ",receptors_per_cell,flush=True, file=logstream)
    print("EC50_number_per_node ",EC50_number_per_node," EC50_bound_in_cycle ",EC50_bound_in_cycle,flush=True, file=logstream)
    print("Hill coefficient ",n_hill,flush=True, file=logstream)
    print("bind_1reaction ",bind_1reaction," useMultiproc ",useMultiprocConstant," useMap ",useMapConstant,flush=True, file=logstream)
    print("invasion_done_perc ",invasion_done_perc,flush=True, file=logstream)
    print("n_runs ", n_runs, " cores2use ",cores2use, flush=True, file=logstream)
    print("bound_PG_offset ",bound_PG_offset," bound_benef_offset ",bound_benef_offset,flush=True, file=logstream)
    print("use_bound_benef_offset ",use_bound_benef_offset," benef_offset ",benef_offset,flush=True, file=logstream)
    print("runs_in_parallel ",runs_in_parallel, flush=True, file=logstream)
    print("benef_frac_out, cycleN_out, boundSum_out ", benef_frac_out, cycleN_out, boundSum_out, flush=True, file=logstream)
    print("GF_sums_list, gf_spat_profile,benef_size_out ",GF_sums_list, gf_spat_profile, benef_size_out, flush=True, file=logstream)
    print("fixedSize, expandDet ",fixedSize, expandDet, flush=True, file=logstream)
    print(f"out2screen {out2screen}, DIFFUSION_DEBUG {DIFFUSION_DEBUG}, BINDING_DEBUG {BINDING_DEBUG}, RNG_DEBUG {RNG_DEBUG}, CC_DEBUG {CC_DEBUG}",flush=True, file=logstream)
    print("Calculated global constants",flush=True, file=logstream)
    print("diff_coef_scaled, diff_delt, k_on_norm, secrete_tstep in main ",
          round(diff_coef_scaled,3), round(diff_delt,3),f'{k_on_norm:.3}', f'{secrete_tstep:.3}',flush=True, file=logstream)
    print("benefFactor ",benefFactor, file=logstream)
    print("\n",flush=True, file=logstream)
    if logstream is not sys.stdout:
        logstream.close()

def cellCycleLoop(local_rng,boundbenefarray,gfcoords,cellarray):
    """
    06/07/22 Main worker sub called after all initializations
             takes initial bound benefit lattice-style array (benefarray)
             and initial matrix of free GF coords (gfcoords), as well as
             lattice of cells (cellarray)
             Does diffusion-binding ("stepping") -> cell proliferation (EGT)
    07/19/22 Changed for parallel running; now output (invasion_ctr,perishing_ctr)
    06/26/23 Changing for 3D
    09/29/23 Changing for 1d
    """
    global n_cycles, cellcycle_t, secrete_tstep, L
    global EC50_number_per_node, n_hill
    global invasion_done_perc, useCost, cost
    global benefFactor
    global df_spatial_free_gf,df_spatial_bound_gf
    global benef_offset
    global fixedSize, expandDet
    global logfile
    #expandingSize = not fixedSize
    if CC_DEBUG:
          if logfile == "__stdout__":
              logstream = sys.stdout
          else:
              logstream = open(logfile,'a')
                    
    secret_per_cycle = round(cellcycle_t/secrete_tstep)
    if CC_DEBUG:
        print("Process: ", mp.current_process()," secret_per_cycle ",secret_per_cycle,flush=True, file=logstream)
    
    # looping over n_cycles cell cycles, quit earlier if condition
    # of invasion fulfilled 
    #for cycleN in range(n_cycles):
    #print("initial cell array: ",cellarray,flush=True)
    for cycleN in range(n_cycles):
        # within 1 cell cycle
        if CC_DEBUG:
            print("Process: ", mp.current_process()," cycle ",cycleN, file=logstream)
        #print("Process: ", mp.current_process()," cycle ",cycleN,flush=True)
        # looping over secretions
        #starttime=0
        for secrStep in range(secret_per_cycle):
            #if secrStep%200==0:
            #    endtime = timer()
            #    print("end-start: ",endtime-starttime)
            #    starttime=timer()
            #    print("Process: ", mp.current_process()," secrStep ",secrStep)
            # perform diffusion-binding
            #print("before doStepping\n",gfcoords)
            (boundbenefarray,gfcoords) = doStepping(local_rng=local_rng,
                                                    benefarray= boundbenefarray,
                                                    gfcoords  = gfcoords,
                                                    cellarray = cellarray
                                                    )
            #print("after doStepping\n",gfcoords)
            # next secretion
            #print("before secretegf")
            gfcoords = secreteGF(cellarray,oldGfDf=gfcoords)
            #print("number of bound GFs", np.sum(boundbenefarray))

        # finding benefits for each cell
        #print("after secretion loop")
        boundbenefarray_1d = boundbenefarray
        resbenefarray = np.array(list(map(hillFn,boundbenefarray_1d,
                                          [EC50_number_per_node]*L,
                                          [n_hill]*L,
                                          [benef_offset]*L)),dtype="float")
        if benef_size_out:
           max_benef_list.append(np.max(resbenefarray))
           min_benef_list.append(np.min(resbenefarray))
        #print("after hillFn")
        #print("benefarray\n")
        #print(boundbenefarray)
        #print("cellbenefarray\n")
        #print(type(resbenefarray))
        #print(resbenefarray.ndim)
        #print(resbenefarray.shape)
        #print("resbenefarray \n",resbenefarray)
        #print("max(resbenefarray) \n",np.max(resbenefarray))
        #print("min(resbenefarray) \n",np.min(resbenefarray))
        
        # if using costs, reduce benefits of "2"s
        if useCost:
            cellarray_1d = cellarray
            resbenefarray[cellarray_1d==2] = resbenefarray[cellarray_1d==2] - cost
            resbenefarray[resbenefarray<0] = 0
        #########################
        ### before proliferation, calculate fractions of PGs bound by 2's and 1's to later fill histos
        if benef_frac_out or  boundSum_out or GF_sums_list or ((not fixedSize)and(not expandDet)):
            total_bound_sum = np.sum(boundbenefarray)        ## total number of bound
        if benef_frac_out:
            prod_bound = boundbenefarray[cellarray==2]       ## producer-bound array
            prod_bound_sum = np.sum(prod_bound)
            nonprod_bound = boundbenefarray[cellarray==1]	 ## nonproducer-bound array
            nonprod_bound_sum = np.sum(nonprod_bound)
            prod_bound_frac = prod_bound_sum/total_bound_sum
            nonprod_bound_frac = nonprod_bound_sum/total_bound_sum
            prod_bound_frac_list.append(prod_bound_frac)
            nonprod_bound_frac_list.append(nonprod_bound_frac)
        #########################
        
        ## Unlike the simple Expanding dir, increase size not by 1 but proportionally to accumulated
        ## fitness - not L = L+1, but L = L + f(fitness)
        ## total_bound_sum found already, need to scale it to relatively small size increase
        if boundSum_out:
            boundSumList.append(total_bound_sum)
        if CC_DEBUG:
          print("lattice side initial ",L, file=logstream)
        if (not fixedSize):
            if expandDet:
                expansionSize = 1
            else:
                expansionSize = round(total_bound_sum/benefFactor)
                #print("total_bound_sum ",total_bound_sum)
            #print("expansionSize  ",expansionSize)
            L = L + expansionSize
        ## filling lists of bound and unbound GFs        
        if(GF_sums_list):
            bound_sums_list.append(total_bound_sum)
            unbound_sums_list.append(len(gfcoords.index))
        ##
        if gf_spat_profile:
            ### Creating spatial profile of free and bound GFs
            ## free
            df_spatial_free_gf = getSpatialProfile(gfcoords_df = gfcoords, initial_spatial=df_spatial_free_gf)
            ## bound
            df_spatial_bound_gf = getSpatialProfile(gfcoords_df = mat_to_coord1d(boundbenefarray), initial_spatial=df_spatial_bound_gf)
        #print("Bound after getSpatialProfile \n",df_spatial_bound_gf)
        ###
        # proliferation - game theory like "strategy exchange"
        # 10/06/22 added expanding and newL arguments to allow for expanding lattice
        if CC_DEBUG:
          print("lattice side before WF ",L, file=logstream)
        cellarray = wrightFisherSpatial1D(cellarray=cellarray,fitnesses=resbenefarray,rng_arg=local_rng,
                                          expanding=(not fixedSize), newL=L)
        #print("cell array after WF: ",cellarray,flush=True)
        #print("Process: ", mp.current_process()," cellarray in loop 2\n",cellarray,flush=True)

        """
        print("boundbenefarray ",boundbenefarray)
        print("cellarray ",cellarray)
        print("total_bound_sum ",total_bound_sum)
        print("prod_bound ",prod_bound); print("prod_bound_sum ",prod_bound_sum)        
        print("nonprod_bound ",nonprod_bound); print("nonprod_bound_sum ",nonprod_bound_sum)
        print("prod_bound_frac ",prod_bound_frac); print("nonprod_bound_frac ",nonprod_bound_frac)
        print("prod_bound_frac_list ",prod_bound_frac_list); print("nonprod_bound_frac_list ",nonprod_bound_frac_list)        
        input("Press Enter to continue...")
        """
        #print("boundbenefarray before re-ini",boundbenefarray)
        # re-initialize bound benefit array
        boundbenefarray = iniBenefitArray()
        """
        print("L ",L)
        print("cellarray ",cellarray)
        print("boundbenefarray after re-ini",boundbenefarray)
        print("gfcoords ",gfcoords)
        #input("Press Enter to continue...")
        """
        
        # checking if invasion condition fulfilled
        prod_number = np.count_nonzero(cellarray==2)
        if prod_number==0:
          if CC_DEBUG:
             print("Process: ", mp.current_process()," PERISHED, exiting",flush=True, file=logstream)
             if logstream is not sys.stdout:
                logstream.close()
          if cycleN_out:
                perish_cycleN.append(cycleN)
          return (0,1)
        non_prod_number = np.count_nonzero(cellarray==1)
        prod_fraction = prod_number/(prod_number+non_prod_number)
        if CC_DEBUG:
          print("Process: ", mp.current_process()," Producer fraction is ",prod_fraction,flush=True, file=logstream)
        if prod_fraction >= invasion_done_perc:
            if CC_DEBUG:
               print("Process: ", mp.current_process()," INVADED, exiting",flush=True, file=logstream)
               if logstream is not sys.stdout:
                  logstream.close()
            if cycleN_out:
                invasion_cycleN.append(cycleN)
            return (1,0)
    if CC_DEBUG: # added fix 11/29/23
        if logstream is not sys.stdout:
            logstream.close()
    return (0,0)

def one_run(local_seed):

    #global logfile
    
    #initializeGlobals()
    printGlobalVars()
    
    #print("Process: ", mp.current_process(), "after global vars", file=logstream)
    
    local_rng = np.random.default_rng(local_seed)
    
    # initialize/setup lattice of producers/non-producers/empty nodes
    #print("Process: ", mp.current_process()," initializing square lattice", file=logstream)
    cellArray = iniLinearLattice()

    # initialize bound benefit lattice (how much benefit is bound in each node)
    # at the moment I do not see how it can be different from all 0's, but won't hurt
    #print("Process: ", mp.current_process()," initializing benefit array", file=logstream)
    boundBenefitArray0 = iniBenefitArray()

    # initialize growth factor secretion; since initialization,
    # old GF dataframe is empty
    #print("Process: ", mp.current_process()," initializing gf array", file=logstream)
    gfCoords0 = secreteGF(cellArray,oldGfDf=pd.DataFrame(columns=['x']))
    #sys.exit(0)
    #print("Process: ", mp.current_process()," cellArray \n",cellArray, flush=True)
    #print("Process: ", mp.current_process()," before cell cycle loop", file=logstream)
    (inv_ctr,perish_ctr) = cellCycleLoop(local_rng=local_rng,
                                         boundbenefarray = boundBenefitArray0,
                                         gfcoords   = gfCoords0,
                                         cellarray  = cellArray)
    #print("Process: ", mp.current_process()," after cellCycleLoop", file=logstream)
    #sys.exit(0)
    return (inv_ctr,perish_ctr)

        
def main(argv):
    global n_runs, cores2use, runs_in_parallel
    global k_on, k_off, diff_coef
    global benef_frac_out, cycleN_out, boundSum_out
    global diff_coef, k_on, k_off, conf_file, cost
    global logfile

    # getting command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--cfg_file", default="def_cfg_gen", help="Configuration file base, without .py extension (default %(default)s)")
    parser.add_argument("--kon", type=float, required=True, help = "Association constant, no default, always required")
    parser.add_argument("--koff", type=float, required=True, help = "Dissociation constant, no default, always required")
    parser.add_argument("-D","--diff", type=float, required=True, help = "Diffusivity, no default, always required")
    parser.add_argument("-c","--cost", type=float, required=True, help = "Cost, no default for safety reasons, always required")    
    args = parser.parse_args()
    conf_file = args.cfg_file; diff_coef = args.diff; k_on = args.kon; k_off = args.koff; cost = args.cost

    print(f"\n\n\n conf_file {conf_file} \n\n\n")
    
    # initialize vars from config file
    initializeGlobals(logfile_arg="",D_arg=diff_coef,kon_arg=k_on,koff_arg=k_off,cost_arg=cost,conf_file=conf_file)
    
    # log file, needs to be after initialization of variables b/c of fixedSize/expandDet used in naming
    today = date.today()
    if out2screen:
        logfile   = "__stdout__"
        logstream = sys.stdout
    else:
        if fixedSize:
            expandStr = "fixed"
        else:
            if expandDet:
                expandStr = "detExp"
            else:
                expandStr =	"propExp"
        logfile = "consol_" + expandStr + "_D_" + "{:.0f}".format(diff_coef) + "_kon_" + "{:.2e}".format(k_on) + \
            "_koff_" + "{:.2e}".format(k_off) + "_cost_" + "{:.2f}".format(cost) + "_" + today.strftime("%m%d%y") + ".log"        
        logstream = open(logfile,'w')

    print(f"Configuration file used: {conf_file}",file=logstream)
    if not useCost:
        print("useCost False, setting cost to 0",file=logstream)
        cost = 0
    if logstream is not sys.stdout:
        logstream.close()

    # printing globals (?put inside DEBUG condition?)
    #printGlobalVars()
    
    ## parallel hanging fix from https://pythonspeed.com/articles/python-multiprocessing/
    mp.set_start_method("spawn")
    # calling one_run in parallel to run for n_runs
    start = timer()

    pool = mp.Pool(processes = cores2use, initializer = initializeGlobals, initargs = (logfile,diff_coef,k_on,k_off,cost,conf_file))
    ss = rng.bit_generator._seed_seq
    child_states = ss.spawn(n_runs)
    resarray = pool.map(one_run,child_states)
    pool.close()

    end = timer()
    if out2screen:
        logstream = sys.stdout
    else:
        logstream = open(logfile,'a')
    print("Process: ", mp.current_process()," time of running: ",end-start,flush=True, file=logstream)
    print("type(resarray) ",type(resarray), file=logstream)
    print("resarray ",resarray, file=logstream)
    
        
    sums = [sum(x) for x in zip(*resarray)]
    invasion_ctr = sums[0]
    perishing_ctr= sums[1]
    
    print("Process: ", mp.current_process()," invasion_ctr ",invasion_ctr," perishing_ctr ",perishing_ctr, file=logstream)

    generic_file_addition = "1D_fixed_"+str(fixedSize)+"_expandDet_"+str(expandDet)+"_D_"+str(round(diff_coef))+"_k_on_"+"{:.2e}".format(k_on)+"_k_off_"+"{:.2e}".format(k_off)+ "_cost_" + "{:.2f}".format(cost) + "_" + today.strftime("%m%d%y")
    
    if benef_frac_out:
        ### histograms of bound/unbound fractions    
        filename_base = "fraction_benef_bound"+generic_file_addition
        filename_pdf  = filename_base+".pdf"
        filename_txt  = filename_base+".txt"
        #plt.savefig("fraction_benef_bound_prod2.pdf")
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        n, bins, patches = ax1.hist(prod_bound_frac_list)
        ax1.title.set_text("Producer bound fraction")
        
        n, bins, patches = ax2.hist(nonprod_bound_frac_list)
        ax2.title.set_text('Non-producer bound fraction')
        
        fig.tight_layout()
        fig.savefig(filename_pdf)
        #
        f1 = open(filename_txt, "w")
        f1.write("prod_bound_frac_list\n")
        f1.write(str(prod_bound_frac_list)+"\n")
        f1.write("nonprod_bound_frac_list\n")    
        f1.write(str(nonprod_bound_frac_list)+"\n")
        f1.close()

    if cycleN_out:
        ### histograms of # of cycles to invasion
        filename_base = "cycleN_at_invasion"+generic_file_addition
        filename_pdf  = filename_base+".pdf"
        filename_txt  = filename_base+".txt"
        fig2 = plt.figure()
        ax3  = fig2.subplots()
        n, bins, patches = ax3.hist(invasion_cycleN)
        ax3.title.set_text("Number of cycles at invasion")
        ax3.set_xlabel("cycle N")
        fig2.savefig(filename_pdf)
        #
        f2 = open(filename_txt, "w")
        f2.write("Cycle number at invasion:\n")
        f2.write(str(invasion_cycleN)+"\n")
        f2.close()
    
        ### histograms of # of cycles to perishing 
        filename_base = "cycleN_at_perishing"+generic_file_addition
        filename_pdf  = filename_base+".pdf"
        filename_txt  = filename_base+".txt"
        fig3 = plt.figure()
        ax4  = fig3.subplots()
        n, bins, patches = ax4.hist(perish_cycleN)
        ax4.title.set_text("Number of cycles at perish")
        ax4.set_xlabel("cycle N")
        fig3.savefig(filename_pdf)
        #
        f3 = open(filename_txt, "w")
        f3.write("Cycle number at perishing:\n")
        f3.write(str(perish_cycleN)+"\n")
        f3.close()

    if boundSum_out:
        ### bound sum histo - temp to figure out range for scaling size_increase
        filename_base = "boundSum"+generic_file_addition
        filename_pdf  = filename_base+".pdf"
        filename_txt  = filename_base+".txt"
        fig4 = plt.figure()
        ax5  = fig4.subplots()
        n, bins, patches = ax5.hist(boundSumList)
        ax5.title.set_text("Bound sum")
        #ax4.set_xlabel("cycle N")
        fig4.savefig(filename_pdf)
        #                                                                                                                                                               
        f4 = open(filename_txt, "w")
        #f4.write("Cycle number at perishing:\n")
        f4.write(str(boundSumList)+"\n")
        f4.close()

    if GF_sums_list:
        filename_base = "GFsums"+generic_file_addition
        filename_pdf  = filename_base+".pdf"
        filename_txt  = filename_base+".txt"
        fig5 = plt.figure()
        ax6 = fig5.add_subplot(2, 1, 1)
        ax7 = fig5.add_subplot(2, 1, 2)
        n, bins, patches = ax6.hist(bound_sums_list)
        ax6.title.set_text("GF bound sums")

        n, bins, patches = ax7.hist(unbound_sums_list)
        ax7.title.set_text("GF unbound sums")        

        fig5.tight_layout()
        fig5.savefig(filename_pdf)
        #                                                                                                                                       
        f5 = open(filename_txt, "w")
        f5.write("bound_sums_list\n")
        f5.write(str(bound_sums_list)+"\n")
        f5.write("unbound_sums_list\n")
        f5.write(str(unbound_sums_list)+"\n")
        f5.close()

    if gf_spat_profile:
        filename_base = "spatial_distr"+generic_file_addition
        filename_pdf  = filename_base+".pdf"
        filename_csv1  = filename_base+"free.csv"
        filename_csv2  = filename_base+"bound.csv"
        fig6, ax8 = plt.subplots(1,2)
        print("df_spatial_free_gf \n",df_spatial_free_gf, file=logstream)
        print("df_spatial_bound_gf \n",df_spatial_bound_gf, file=logstream)
        #sns.barplot(data=df_spatial_free_gf,x=df_spatial_free_gf.index,y="count",ax=ax8[0])
        #sns.barplot(data=df_spatial_bound_gf,x=df_spatial_bound_gf.index,y="count",ax=ax8[1])
        ax8[0].title.set_text("Free GFs")
        ax8[1].title.set_text("Bound GFs")
        fig6.savefig(filename_pdf)
        #
        df_spatial_free_gf.to_csv(filename_csv1)
        df_spatial_bound_gf.to_csv(filename_csv2)

    if benef_size_out:
        filename_base = "benefits"+generic_file_addition
        filename_pdf  = filename_base+".pdf"
        filename_txt  = filename_base+".txt"
        fig7 = plt.figure()
        ax9  = fig7.add_subplot(2, 1, 1)
        n, bins, patches = ax9.hist(max_benef_list)
        ax9.title.set_text("Maximum benefits")

        ax10 = fig7.add_subplot(2, 1, 2)
        n, bins, patches = ax10.hist(min_benef_list)
        ax10.title.set_text("Minimum benefits")

        fig7.savefig(filename_pdf)
        #                                                                                                                                                                                               
        f6 = open(filename_txt, "w")
        f6.write("max_benef_list\n")
        f6.write(str(max_benef_list)+"\n")
        f6.write("min_benef_list\n")
        f6.write(str(min_benef_list)+"\n")
        f6.close()

    
    if RNG_DEBUG and runs_in_parallel: # histos to files
        print("making  rng debug plots", file=logstream)
        fig, ax = plt.subplots()
        plt.hist(rng_proc1)
        plt.text(0.8,0.9,s='mean = {0}'.format(np.mean(rng_proc1),4),transform=ax.transAxes)
        plt.text(0.8,0.8,s='stdev = {0}'.format(np.std(rng_proc1),4),transform=ax.transAxes)        
        plt.savefig("rng_proc1_usemap.pdf")
        #                                                                                                                              
        fig, ax = plt.subplots()
        plt.hist(rng_proc2)
        plt.text(0.8,0.9,s='mean = {0}'.format(np.mean(rng_proc2),4),transform=ax.transAxes)
        plt.text(0.8,0.8,s='stdev = {0}'.format(np.std(rng_proc2),4),transform=ax.transAxes)
        plt.savefig("rng_proc2_usemap.pdf")
        #
        fig, ax = plt.subplots()
        plt.hist(rng_proc3)
        plt.text(0.8,0.9,s='mean = {0}'.format(np.mean(rng_proc3),4),transform=ax.transAxes)
        plt.text(0.8,0.8,s='stdev = {0}'.format(np.std(rng_proc3),4),transform=ax.transAxes)
        plt.savefig("rng_proc3_usemap.pdf")
        #                                                                                                                              
        fig, ax = plt.subplots()
        plt.hist(rng_proc4)
        plt.text(0.8,0.9,s='mean = {0}'.format(np.mean(rng_proc4),4),transform=ax.transAxes)
        plt.text(0.8,0.8,s='stdev = {0}'.format(np.std(rng_proc4),4),transform=ax.transAxes)
        plt.savefig("rng_proc4_usemap.pdf")
        #                                                                                                                              
        fig, ax = plt.subplots()
        plt.hist(rng_proc5)
        plt.text(0.8,0.9,s='mean = {0}'.format(np.mean(rng_proc5),4),transform=ax.transAxes)
        plt.text(0.8,0.8,s='stdev = {0}'.format(np.std(rng_proc5),4),transform=ax.transAxes)
        plt.savefig("rng_proc5_usemap.pdf")

    if logstream is not sys.stdout:
        logstream.close()
        
if __name__=="__main__":
    main(sys.argv[1:])
