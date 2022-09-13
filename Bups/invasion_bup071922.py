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

"""
import sys
import importlib
import numpy as np
from scipy import constants
import pandas as pd
import time
from timeit import default_timer as timer
import multiprocessing as mp

from utilFunctions import coord_to_mat, mat_to_coord
from checkingFunctions import plotGFs, checkDiffusion
from benefitFns        import hillFn
from updating          import wrightFisherSpatial

cores = mp.cpu_count() - 1
rngseed = 50                          # seed for debugging
rng = np.random.default_rng()  # can be called without a seed

DIFFUSION_DEBUG = False
BINDING_DEBUG = False

def initializeGlobals():
    """
    Parameters of the simulation used by different functions: physical/lattice
    constants, time scales
    """
    ################## Constants read in from config file
    global P_mut,Rprod0,inflx,steep,n_cycles,L,L_filled,cellcycle_t
    global secrete_tstep,secreteN,cost,useCost
    global diff_coef, delx, k_on, k_off, receptors_per_cell, um_to_dm
    global EC50_number_per_node, n_hill, EC50_bound_in_cycle
    global bind_1reaction, useMultiprocConstant, useMapConstant
    global updateMethod, invasion_done_perc

    if len(sys.argv)>2:
        sys.exit("Usage: %s [.py config-file basename]" % sys.argv[0])
    elif len(sys.argv)==2:
        conf_file = sys.argv[1]
    else:
        conf_file = "default_cfg_bup071922"
    print("Configuration file used: %s" % conf_file, flush=True)
    try:
        modules = importlib.import_module(conf_file)
        P_mut         = modules.P_mut;         inflx    = modules.inflx
        steep         = modules.steep;         n_cycles = modules.n_cycles;
        L             = modules.L;             L_filled =modules.L_filled;
        cellcycle_t   = modules.cellcycle_t
        secrete_tstep = modules.secrete_tstep; secreteN = modules.secreteN
        cost       = modules.cost;          useCost   = modules.useCost
        diff_coef     = modules.diff_coef;     delx     = modules.delx
        k_on          = modules.k_on;          k_off    = modules.k_off
        receptors_per_cell = modules.receptors_per_cell
        um_to_dm           = modules.um_to_dm
        EC50               = modules.EC50;     n_hill   = modules.n_hill
        bind_1reaction     = modules.bind_1reaction
        useMultiprocConstant = modules.useMultiproc; useMapConstant = modules.useMap
        updateMethod       = modules.updateMethod
        invasion_done_perc = modules.invasion_done_perc
    except ImportError:
        raise ImportError('Cannot import the config file')
    ### Checking input
    assert L>=L_filled, "Filled part can't be larger than lattice"
    ###
    ################## Requiring calculation using read-in constants
    global diff_coef_scaled, diff_delt, k_on_norm

    ### diffusion parameters
    diff_coef_scaled = diff_coef/pow(delx,2) # to use with dx=1 on lattice, delx expected in um => 1/s unit
    #diff_delt = 1/(2*2*diff_coef_scaled)     # diffusion time step, s (see, e.g., "lecture6"); 2nd "2" is dimension of lattice
    diff_delt = 1/(2*2*diff_coef)      # 07/14/22 nothing else is scaled, removing from here (see param write-up) 
    #
    node_volume = pow((delx*um_to_dm),3) # if not mistaken, need "reaction volume" for k_on scaling; converted to dm (k_on has dm^3)
    k_on_norm = k_on/(constants.Avogadro*node_volume) # normalized for Gillespie
    #
    # converting EC50 to number of ligands "around" each node
    EC50_numb_per_liter = EC50 * constants.Avogadro # EC50 in Molar=moles/Liter
    EC50_number_per_node = round(EC50_numb_per_liter * node_volume) # node_volume in dm3=L
    EC50_bound_in_cycle = round(EC50_number_per_node * receptors_per_cell * k_on_norm * cellcycle_t) #reaction rate*time of cell cycle
    
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

def bindingStep(benefarray, gfcoords, cellarray, useMultiproc=True, useMap=True):
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
    
    # 04/21/22 here call the routine running binding procedure for each node
    # expect "+1" if new association at the node, "-1" if new dissociation
    # at the node, "0" if nothing changes
    # ? makes it possibly to run in lapply style later and return a DF ?
    # ? with possible parallel processing ?

    # converting to "lattice matrix"
    coordmat = coord_to_mat(gfcoords.to_numpy(dtype="int"),L)

    #print("Before gillespie: benefarray[3,3] ",benefarray[3,3]," coordmat[3,3] ",coordmat[3,3])
    #boundben, freeben = GillespieFn(benefBound=benefarray[3,3],benefFree=coordmat[3,3])
    #print("After gillespie: boundben ",boundben," freeben ",freeben)

    # bind_1reaction defines if only 1 possible reaction per binding
    # step is considered - yes/no, then exit; if false, then after
    # 1 reaction occured, probability of the next one before the step
    # time is over is checked - quite unlikely for parameters used on
    # 05/17/22 (vegf stuff) and slows down a lot

    benefarray_1d = benefarray.flatten(); coordmat_1d = coordmat.flatten()
    cellarray_1d = cellarray.flatten()
    #start = timer()
    #rands0 = rng.random(size=L*L); rands1 = rng.random(size=L*L)
    #end = timer()
    #print("Time of generation: ",end-start)
    if bind_1reaction:
        rands0 = rng.random(size=L*L); rands1 = rng.random(size=L*L)
        if useMultiproc:
            pool = mp.Pool(cores)
            resarray = np.array(pool.starmap(GillespieFn_1react,zip(benefarray_1d,coordmat_1d,cellarray_1d,rands0,rands1)),dtype="int")
            pool.close()
        elif useMap:
            resarray = np.array(list(map(GillespieFn_1react,benefarray_1d,coordmat_1d,cellarray_1d,rands0,rands1)),dtype="int")
        else:
            resarray = np.array([GillespieFn_1react(x,y,z,a) for x,y,z,a in zip(benefarray_1d,coordmat_1d,cellarray_1d,rands0,rands1)],dtype="int")
    else:
        if useMultiproc: # b/c of RNG, had to code a separate routine
            pool = mp.Pool(cores)
            ss = rng.bit_generator._seed_seq
            child_states = ss.spawn(L*L)
            resarray = np.array(pool.starmap(GillespieFn_multi,zip(benefarray_1d,coordmat_1d,cellarray_1d,child_states)),dtype="int")
            pool.close()
        elif useMap:
            resarray = np.array(list(map(GillespieFn,benefarray_1d,cellarray_1d,coordmat_1d)),dtype="int")
        else:
            resarray = np.array([GillespieFn(x,y,z) for x,y,z in zip(benefarray_1d,coordmat_1d,cellarray_1d)],dtype="int")
    

    benefarray_1d = resarray[:,0]; benefarray = benefarray_1d.reshape(L,L)
    coordmat_1d = resarray[:,1];   coordmat   = coordmat_1d.reshape(L,L)

    # converting back to dataframe
    gfcoords = mat_to_coord(coordmat)

    
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
        if rand1*a0 <= a1:
            if BINDING_DEBUG:
                print("\nASSOCIATION\n")
                print("rand0 ",rand0," rand1 ",rand1)
                print("a1 ",a1," a2 ",a2," tau ",tau) 
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
    if rand1*a0 <= a1:
        benefBound = benefBound + 1
        benefFree  = benefFree  - 1
        if BINDING_DEBUG:
            print("\nASSOCIATION\n")
            print("rand0 ",rand0," rand1 ",rand1)
            print("a1 ",a1," a2 ",a2," tau ",tau)         
            print("benefBound ",benefBound)
            print("benefFree ",benefFree)
    else:
        benefBound = benefBound - 1
        benefFree  = benefFree  + 1
        
    return (benefBound,benefFree)
        
def GillespieFn(benefBound,benefFree,cell):#,rand0,rand1):
    """
    Incorporating Gillespie routine, k_on normalized a la Gabhann/Popel
    Input: benefBound - number of "goodies" already bound (b4 this step) by node/cell,
                        can dissociate, benefarray or boundBenefitArray elsewhere
           benefFree  - number of "goodies" that just diffused here, can bind
                        gfcoords or coordmat elsewhere
    Output: 1) benefBound, 2) benefFree after the step
    """
    global k_on_norm, k_off, diff_delt, receptors_per_cell
    
    print("benefBound ",benefBound," benefFree ",benefFree," cell ",cell)

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
    while 1:
        # 1) calculating "a's" (see paper), a1 - association, a2 - dissociation
        a1 = benefFree*receptors_per_cell*k_on_norm
        a2 = k_off * benefBound
        a0 = a1 + a2

        # 2) generating random numbers, finding time/reaction
        rands = rng.random(size=2); rand0=rands[0]; rand1=rands[1]
        #rand0,rand1 = rng.random(size=2)
        #print("rands")
        #print(rands)
        #rands[0] = 0.98
        tau = (1/a0)*np.log(1/rand0)
        curr_step_time = curr_step_time + tau        
        if(curr_step_time>diff_delt):
            return (benefBound,benefFree)
        if rand1*a0 <= a1:
            if BINDING_DEBUG:
                print("\nASSOCIATION\n")
                print("rand0 ",rand0," rand1 ",rand1)
                print("a1 ",a1," a2 ",a2," tau ",tau) 
            benefBound = benefBound + 1
            benefFree  = benefFree  - 1
        else:
            benefBound = benefBound - 1
            benefFree  = benefFree  + 1
    
    

def doStepping(benefarray,
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
    global secrete_tstep, diff_delt
    global useMultiprocConstant, useMapConstant # using || running or map/list comprehension
    
    # satisfying the last condition by calculating max step number as the number
    # of diffusion steps fitting in the secretion delt
    # secrete_delt expected in s, in config file
    # diff_delt should be seconds, see comment to calculating it and above
    Nsteps_max = round(secrete_tstep/diff_delt)
    #Nsteps_max=1
    if DIFFUSION_DEBUG:
        print("Nsteps_max ",Nsteps_max)
        xy_df_to_plot = pd.DataFrame({'x':[],'y':[],'step':[]},dtype=int)
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
        gfcoords = RWstep(gfcoords)
        if DIFFUSION_DEBUG and (stepN in steps2plot): # preparing to plot FG coords for some steps if debugging
            stepdata = pd.DataFrame({'x':gfcoords.x,'y':gfcoords.y, 'step':[stepN]*len(gfcoords.x)})            
            xy_df_to_plot = xy_df_to_plot.append(stepdata,ignore_index=True)
            
        ## check if any GF's left on lattice (condition 2)
        if(len(gfcoords)==0):            
            #print("No GFs left on lattice condition triggered")
            return (benefarray,gfcoords)
        
        ## binding (or not, to cell to which drifted)
        #print(stepN)
        (benefarray, gfcoords) = bindingStep(benefarray, gfcoords, cellarray,
                                             useMultiproc=useMultiprocConstant, useMap=useMapConstant)
        #(benefarray, gfcoords) = bindingStep_mp(benefarray, gfcoords)
        """
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
        """
    if DIFFUSION_DEBUG: # plotting prepared GF coords if debugging
        xy_df_to_plot = xy_df_to_plot.apply(pd.to_numeric, errors='coerce')# get rid of "object" types
        xy_df_to_plot = xy_df_to_plot.astype("int",errors="ignore")        # step was still "float"
        print(xy_df_to_plot)
        plotGFs(xy_df_to_plot)

    return benefarray, gfcoords
        
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

def printGlobalVars():
    # global vars from config file
    global P_mut,Rprod0,inflx,steep,n_cycles,L,L_filled,cellcycle_t
    global secrete_tstep,secreteN,cost, useCost
    global diff_coef, delx, k_on, k_off, receptors_per_cell
    global EC50_number_per_node, n_hill, EC50_bound_in_cycle
    global bind_1reaction, useMultiprocConstant, useMapConstant
    global updateMethod, invasion_done_perc
    # global vars not from config file
    global diff_coef_scaled, diff_delt, k_on_norm
    print("\n",flush=True)
    print("Constants from the configuration file",flush=True)
    print("P_mut,inflx,steep,n_cycles,L,L_filled in main: ",P_mut,inflx,steep,n_cycles,L,L_filled,flush=True)
    print("cellcycle_t, secrete_tstep, secreteN: ", cellcycle_t, secrete_tstep, secreteN,flush=True)
    print("cost, useCost: ",cost, useCost, flush=True)
    print("diff_coef, delx, k_on, k_off in main ", round(diff_coef,3), delx, k_on, k_off,flush=True)
    print("receptors_per_cell in main ",receptors_per_cell,flush=True)
    print("EC50_number_per_node ",EC50_number_per_node," EC50_bound_in_cycle ",EC50_bound_in_cycle,flush=True)
    print("Hill coefficient ",n_hill,flush=True)
    print("bind_1reaction ",bind_1reaction," useMultiproc ",useMultiprocConstant," useMap ",useMapConstant,flush=True)
    print("updateMethod ",updateMethod,flush=True)
    print("invasion_done_perc ",invasion_done_perc,flush=True)
    print("Calculated global constants",flush=True)
    print("diff_coef_scaled, diff_delt, k_on_norm in main ",
          round(diff_coef_scaled,3), round(diff_delt,3),f'{k_on_norm:.3}',flush=True)
    print("\n",flush=True)

def cellCycleLoop(boundbenefarray,gfcoords,cellarray):
    """
    06/07/22 Main worker sub called after all initializations
             takes initial bound benefit lattice-style array (benefarray)
             and initial matrix of free GF coords (gfcoords), as well as
             lattice of cells (cellarray)
             Does diffusion-binding ("stepping") -> cell proliferation (EGT)             
    """
    global n_cycles, cellcycle_t, secrete_tstep, L
    global EC50_number_per_node, n_hill
    global invasion_done_perc, useCost, cost

    secret_per_cycle = round(cellcycle_t/secrete_tstep)
    print("secret_per_cycle ",secret_per_cycle,flush=True)
    
    # looping over n_cycles cell cycles, quit earlier if condition
    # of invasion fulfilled 
    #for cycleN in range(n_cycles):
    for cycleN in range(n_cycles):
        # within 1 cell cycle
        print("cycle ",cycleN,flush=True)
        # looping over secretions
        for secrStep in range(secret_per_cycle):
        #for secrStep in range(300):
            #if secrStep%100==0:
            #    print("secrStep ",secrStep)
            # perform diffusion-binding
            (boundbenefarray,gfcoords) = doStepping(benefarray= boundbenefarray,
                                                    gfcoords  = gfcoords,
                                                    cellarray = cellarray
                                                    )
            # next secretion
            gfcoords = secreteGF(cellarray,oldGfDf=gfcoords)
            #print("number of bound GFs", np.sum(boundbenefarray))

        # finding benefits for each cell
        boundbenefarray_1d = boundbenefarray.flatten()    
        resbenefarray = np.array(list(map(hillFn,boundbenefarray_1d,
                                          [EC50_number_per_node]*L*L,
                                          [n_hill]*L*L)),dtype="float")
                
        #print("benefarray\n")
        #print(boundbenefarray)
        #print("cellbenefarray\n")
        #print(type(resbenefarray))
        #print(resbenefarray.ndim)
        #print(resbenefarray.shape)
        #print(resbenefarray)        
        
        # if using costs, reduce benefits of "2"s
        if useCost:
            cellarray_1d = cellarray.flatten()
            resbenefarray[cellarray_1d==2] = resbenefarray[cellarray_1d==2] - cost
            resbenefarray[resbenefarray<0] = 0

        print("cellarray")
        print(cellarray)
            
        # proliferation - game theory like "strategy exchange"
        cellarray = wrightFisherSpatial(cellarray=cellarray,fitnesses=resbenefarray,rng_arg=rng)
        print("cellarray in loop",flush=True)
        print(cellarray,flush=True)

        # re-initialize bound benefit array
        boundbenefarray = iniBenefitArray()

        # checking if invasion condition fulfilled
        prod_number = np.count_nonzero(cellarray==2)
        non_prod_number = np.count_nonzero(cellarray==1)
        prod_fraction = prod_number/(prod_number+non_prod_number)
        print("Producer fraction is ",prod_fraction,flush=True)
        if prod_fraction >= invasion_done_perc:
            print("INVADED, exiting",flush=True)
            return

def main():
    global updateMethod
    # initialize vars from config file
    initializeGlobals()

    # printing globals (?put inside DEBUG condition?)
    printGlobalVars()
    
    # initialize/setup lattice of producers/non-producers/empty nodes
    cellArray = iniSquareLattice()

    # initialize bound benefit lattice (how much benefit is bound in each node)
    # at the moment I do not see how it can be different from all 0's, but won't hurt
    boundBenefitArray0 = iniBenefitArray()

    #cellArray = wrightFisherSpatial(cellarray=cellArray,fitnesses=boundBenefitArray0,rng_arg=rng)
    #return 0
    
    # initialize growth factor secretion; since initialization,
    # old GF dataframe is empty
    gfCoords0 = secreteGF(cellArray,oldGfDf=pd.DataFrame(columns=['x','y']))

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

    print("cellArray")
    print(cellArray)
    cellCycleLoop(boundbenefarray = boundBenefitArray0,
                  gfcoords   = gfCoords0,
                  cellarray  = cellArray)
    """
    boundBenefitArray = doStepping(benefarray=boundBenefitArray,
                                   gfcoords=gfCoords
                                   )
    """    
    end = timer()
    print("Time of doStepping: ",end-start,flush=True)
    

if __name__=="__main__":
    main()
