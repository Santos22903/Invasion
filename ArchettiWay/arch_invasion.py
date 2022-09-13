#!/usr/bin/env python3
"""
Author: Svyatoslav Tkachenko
Date:   06/28/22

Simulation of cells producing a public benefit
growth factor invading a population of 
non-producers done in the Archetti approach
(M.Archetti, PLoS, 2014, 9,9)

07/07/22 Backed up to switch from Euclidean to Manhattan metric
         (from Archetti's formula for group size looks like he use it+
          it is only logical on the lattice)
07/12/22 Adding "wrap around to toroid" version of finding cell contributions
         The difference in: when finding distances to "2"s - get distance
         to closest including along the back side of toroid.
         Change needed only in manhattan_distance_2d - will keep old version
         + add  manhattan_distance_2d_wrap and flag in config file - which
         version to use
"""

import sys
import importlib
import numpy     as np
import math
from timeit import default_timer as timer

rngseed = 50                          # seed for debugging
rng = np.random.default_rng()  # can be called without a seed

def initializeGlobals():
    """
    Parameters of the simulation used by different functions: physical/lattice
    constants, time scales
    """
    ################## Constants read in from config file
    global L, L_filled, diff_range
    global dParam, zParam
    global cost, useCost, inflection, steepness
    global update_type, n_cycles, invasion_done_perc
    global n_runs, findContrib

    if len(sys.argv)>2:
        sys.exit("Usage: %s [.py config-file basename]" % sys.argv[0])
    elif len(sys.argv)==2:
        conf_file = sys.argv[1]
    else:
        conf_file = "default_cfg"
    print("Configuration file used: %s" % conf_file)
    try:
        modules = importlib.import_module(conf_file)
        L          = modules.L;             L_filled  = modules.L_filled
        diff_range = modules.diff_range
        dParam     = modules.dParam;        zParam    = modules.zParam
        cost       = modules.cost;          useCost   = modules.useCost
        inflection = modules.inflection;    steepness = modules.steepness
        update_type= modules.update_type;   n_cycles  = modules.n_cycles
        invasion_done_perc = modules.invasion_done_perc
        n_runs     = modules.n_runs;        findContrib = modules.findContrib
    except ImportError:
        raise ImportError('Cannot import the config file')
    ### Checking input
    assert L>=L_filled, "Filled part can't be larger than lattice"
    assert update_type in ["deterministic","stochastic"], "update type should be either 'deterministic' or 'stochastic'"
    assert findContrib in ["wrap","direct"], "finding contribution should be either 'wrap' or 'direct'"
    ###
    ################## Requiring calculation using read-in constants
    global group_size
    group_size = 2*diff_range*(diff_range+1) + 1
    
    return

def printGlobalVars():
    # global vars from config file
    global L, L_filled, diff_range
    global dParam, zParam
    global cost, useCost, inflection, steepness
    global update_type, n_cycles, invasion_done_perc
    global n_runs, findContrib

    print("\n")
    print("Constants from the configuration file")
    print("L, L_filled, diff_range in main: ", L, L_filled, diff_range)
    print("dParam, zParam in main: ", dParam, zParam)
    print("cost, useCost, inflection, steepness in main: ",cost, useCost, inflection, steepness)
    print("update_type, invasion_done_perc ",update_type,invasion_done_perc)
    print("n_runs, findContrib ",n_runs,findContrib)

    print("\n")
    print("Calculated constants")
    print("group_size ",group_size)

    print("\n")

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
    #iniLattice[0,0]=2 # for debugging
    """
    iniLattice[1,1]=2
    iniLattice[2,2]=2
    iniLattice[3,3]=2
    iniLattice[4,4]=2
    iniLattice[5,5]=2
    #iniLattice[11,11]=2
    #iniLattice[11,13]=2
    """

    return iniLattice

def findCellContributions(cellarray):
    """
    Finds number of contributors to each cell
    Input:  array of cells
    Output: array (same size) of contributor numbers for each cell
    """
    global L, findContrib
    ### array of "2" coordinates
    coord_2_tuple = np.where(cellarray == 2)
    coord_2_array = np.asarray(coord_2_tuple).T
    #print(coord_2_array)

    ### calculate distances to "2" for all + filter by diff_range
    ### get 3d array, where each entry in 2d is a 1d array of distances

    ## make array of coordinates (https://stackoverflow.com/questions/38173572/python-numpy-generate-coordinates-for-x-and-y-values-in-a-certain-range)
    Xpts = np.arange(L)
    Ypts = np.arange(L)
    X2D,Y2D = np.meshgrid(Xpts,Ypts)
    coords  = np.column_stack((X2D.ravel(),Y2D.ravel()))
    #print(coords)

    ## distances as list, all in "j" compared to each in "i", then next in "j"
    ## (this way get distances of all "2"s to each coordinate) => list of lists

    #distsEuclid = [[np.linalg.norm(i-j) for i in coord_2_array] for j in coords]
    #print("distsEuclid")
    #print(distsEuclid)

    if findContrib=="direct":
        distsManhat = [[manhattan_distance_2d(i,j) for i in coord_2_array] for j in coords]
    elif findContrib=="wrap":
        distsManhat = [[manhattan_distance_2d_wrap(i,j) for i in coord_2_array] for j in coords]
    else:
        raise ValueError("findContrib must be 'wrap' or 'direct'")
    #print("distsManhat")
    #print(distsManhat)
    """
    print("type(dists) ",type(dists))
    print("type(dists[0]) ",type(dists[0]))
    print("type(dists[0][0]) ",type(dists[0][0]))
    print("dists[0][0] ",dists[0][0])
    print("dists[0][1] ",dists[0][1])        
    """
    ### run normContribn over each list + sum over => array of contributions
    ## map function for each cell (running over "inner" list of distances => contribution)
    ## to the big ("outer") list
    contribList = list(map(contribForCell,distsManhat))
    #print(contribList)

    contribArray = np.array(contribList).reshape(L,L)
    return contribArray

def manhattan_distance_2d(vec1,vec2):
    """
    07/07/22
    calculating manhattan distance b/n 2-component vectors
    """
    #print(type(vec1)); print(vec1)
    #print(type(vec2)); print(vec2)
    dist = abs(vec1[0]-vec2[0]) + abs(vec1[1]-vec2[1])
    #print("manhattan distance: ",dist)
    return dist

def manhattan_distance_2d_wrap(vec1,vec2):
    """
    07/12/22
    calculating manhattan distance b/n 2-component vectors,
    the difference with manhattan_distance_2d is that the
    closest distance - either straight or along the back of
    the closed toroid (see Archetti paper) will be returned
    """

    ### to impose periodic conditions, find coord diff'ce as
    ### before, then add "L" to the smaller of each of x,y,
    ### calculate again, choose the smaller for each coord
    xmin = min(vec1[0],vec2[0]); xmax = max(vec1[0],vec2[0])
    ymin = min(vec1[1],vec2[1]); ymax = max(vec1[1],vec2[1])
    #
    xdiff1 = abs(xmax-xmin); xdiff2 = abs(xmax - xmin -L)
    ydiff1 = abs(ymax-ymin); ydiff2 = abs(ymax - ymin -L)
    #
    xdiff = min(xdiff1,xdiff2)
    ydiff = min(ydiff1,ydiff2)
    #
    dist = xdiff + ydiff
    """
    print("vec1: ",vec1); print("vec2: ",vec2)    
    print("xdiff1 ",xdiff1," xdiff2 ",xdiff2," xdiff ",xdiff)
    print("ydiff1 ",ydiff1," ydiff2 ",ydiff2," ydiff ",ydiff)    
    print("manhattan distance: ",dist)
    """
    return dist

def contribForCell(distInnerList):
    #print("distInnerList: ",distInnerList)
    #nodeContribs = list(map(testFn,distInnerList))
    nodeContribs = list(map(normContribFn,distInnerList))
    #print("nodeContribs: ",nodeContribs)
    nodeContribsSum = sum(nodeContribs)
    return nodeContribsSum

def testFn(inpar):
    return inpar*2
    
def contribFn(dist):
    """
    Function finding contribution from each producer before normalization
    """
    global zParam, dParam, diff_range

    return 1/(1+math.exp(-zParam*(dist-dParam)/diff_range))

def normContribFn(distance):
    """
    Function finding normalized contribution from each producer
    """
    global diff_range

    if distance>diff_range:
        return 0
    
    g_0 = contribFn(dist = 0)
    g_D = contribFn(dist = diff_range)
    g_i = contribFn(dist = distance)

    return 1-(g_i-g_0)/(g_D-g_0)

def findCellBenefits(contribarray):
    """
    finding benefits using contribution array (input)
    by scaling it by group size and the running normalized
    benefit function over it (flatten, then map)
    """
    #print("Finding cell benefits")
    global group_size, L

    weighted_fract_array = contribarray/group_size
    #print(weighted_fract_array.round(decimals=3))
    weighted_fract_array_1d = weighted_fract_array.flatten()
    benefitList = list(map(normBenefitFn,weighted_fract_array_1d))
    benefitArray = np.array(benefitList).reshape(L,L)
        
    return benefitArray

def normBenefitFn(weighted_fract):
    """
    normalized benefits for each cell depending
    on weighted fraction of producers in neighb
    """
    b_0 = benefitFn(0)
    b_1 = benefitFn(1)
    b_x = benefitFn(weighted_fract)

    return (b_x-b_0)/(b_1-b_0)

def benefitFn(x):
    """
    logistic f'n (as in Archetti) returning cell
    benefit as a f'n of weighted fract of prod's
    """
    global inflection, steepness
    #print("x: ",x)
    return 1/(1+math.exp(steepness*(inflection-x)))

def strategyUpdate(cellarray,benefitarray):
    """
    Archetti strategy update (p.4 of 2014 paper on heterogeneity)
    A cell (x) is randomly selected, then one of its neighbors (y)
    randomly selected. Benefits of x and y are compared and update
    if needed performed:
    Deterministic: if Px<Py, x adopts y's strategy, otherwise nothing
    Stochastic: strategy replacement with probability (Py-Px)/M,
                where M is the maximum possible Py-Px
    """
    global update_type, L

    ## arrays to choose from
    poss_coords = range(L)
    poss_axis   = [0,1]
    poss_dirs   = [-1,1]
    
    ## choose node to test
    test_node = rng.choice(poss_coords,size=2)
    #print("test_node ",test_node)
    
    ## choose one of its neighbors as a "role model"
    model_axis = rng.choice(poss_axis)
    model_dir  = rng.choice(poss_dirs)
    #print("model_axis ",model_axis," model_dir ",model_dir)
    model_node = np.copy(test_node)
    model_node[model_axis] = model_node[model_axis] + model_dir
    ####### in case of "overflow" - "roll over" ##########
    if model_node[model_axis]==L:
        model_node[model_axis] = 0
    if model_node[model_axis]<0:
        model_node[model_axis] = L-1
    ######################################################
    #print("model_node ",model_node)
    #print("test_node ",test_node)

    ## compare benefits and possibly update (depending on update type)
    Px = benefitarray[test_node[0],test_node[1]]
    Py = benefitarray[model_node[0],model_node[1]]
    #print("Px ",Px," Py ",Py)
    #print(cellarray)
    if update_type=="deterministic":
        if Py>Px:
            cellarray[test_node[0],test_node[1]]=cellarray[model_node[0],model_node[1]]
    elif update_type=="stochastic":        
        if(Py>Px):
            maxdiff = np.amax(benefitarray) - np.amin(benefitarray)
            probab  = (Py-Px)/maxdiff
            testp = rng.uniform(0,1)
            #print("probab ",probab," testp ",testp)
            if testp<=probab: #checking probab>0 redundant - same as Py>Px
                cellarray[test_node[0],test_node[1]]=cellarray[model_node[0],model_node[1]]
                #print(cellarray)
    else:
      raise ValueError("update type must be 'stochastic' or 'deterministic'")          
        
    #print(cellarray)
    return cellarray


def main():
    global n_cycles, invasion_done_perc, n_runs, useCost, cost

    ### initializations
    initializeGlobals()            # global vars from config file
    printGlobalVars()
    
    invasion_ctr = 0
    perishing_ctr= 0

    start = timer()
    
    for runNumber in range(n_runs):
        if runNumber%5 == 0:
            print("runNumber ",runNumber)

        cellArray = iniSquareLattice() # lattice
        #print(cellArray)

        for cycleN in range(n_cycles):
            if cycleN%10000 == 0:
                print("cycleN ",cycleN)
                print("cellArray:")
                print(cellArray)
            ### calculate benefits for each cell
            contribArray = findCellContributions(cellArray) # returns square array of contributors
            #print("contribution: ")
            #print(contribArray.round(decimals=2))
    
            ### play the game as described in the paper

            ## Benefits
            # p.3: weighted number of producers <- sum of all contributions
            # benefit f'n = f(sum of contribs/group size)
            benefitArray = findCellBenefits(contribArray)
            print("Before cost")
            print(benefitArray)
            print("cellArray")
            print(cellArray)
            #print(type(benefitArray))
            #print(benefitArray.shape)

            ### in case cost is used, offset the benefits of "2" by it
            ### if negative, set to 0
            if useCost:
                benefitArray[cellArray==2] = benefitArray[cellArray==2] - cost
                #print("After cost")
                #print(benefitArray)
                benefitArray[benefitArray<0] = 0
                #print("After zeroing")
                #print(benefitArray)
            ## birth - death game
            cellArray = strategyUpdate(cellArray,benefitArray)
            
            # checking if invasion condition fulfilled
            prod_number = np.count_nonzero(cellArray==2)
            if prod_number == 0:
                print("cycleN ",cycleN)
                print("INVASION FAILED, no 2 left")
                print(cellArray)
                perishing_ctr += 1
                break
            non_prod_number = np.count_nonzero(cellArray==1)
            prod_fraction = prod_number/(prod_number+non_prod_number)        
            if prod_fraction >= invasion_done_perc:
                print("cycleN ",cycleN)
                print("INVADED, exiting")
                print(cellArray)
                invasion_ctr += 1
                break
            
    end = timer()
    print("Time of all runs: ",end-start)
    print("invasion_ctr ",invasion_ctr," perishing_ctr ",perishing_ctr)


if __name__=="__main__":
    main()
