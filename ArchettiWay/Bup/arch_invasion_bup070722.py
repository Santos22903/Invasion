#!/usr/bin/env python3
"""
Author: Svyatoslav Tkachenko
Date:   06/28/22

Simulation of cells producing a public benefit
growth factor invading a population of 
non-producers done in the Archetti approach
(M.Archetti, PLoS, 2014, 9,9)
"""

import sys
import importlib
import numpy     as np
import math

def initializeGlobals():
    """
    Parameters of the simulation used by different functions: physical/lattice
    constants, time scales
    """
    ################## Constants read in from config file
    global L, L_filled, diff_range
    global dParam, zParam

    if len(sys.argv)>2:
        sys.exit("Usage: %s [.py config-file basename]" % sys.argv[0])
    elif len(sys.argv)==2:
        conf_file = sys.argv[1]
    else:
        conf_file = "default_cfg"
    print("Configuration file used: %s" % conf_file)
    try:
        modules = importlib.import_module(conf_file)
        L          = modules.L;             L_filled =modules.L_filled
        diff_range = modules.diff_range
        dParam     = modules.dParam;        zParam     = modules.zParam   
    except ImportError:
        raise ImportError('Cannot import the config file')
    ### Checking input
    assert L>=L_filled, "Filled part can't be larger than lattice"
    ###
    ################## Requiring calculation using read-in constants
    
    return

def printGlobalVars():
    # global vars from config file
    global L, L_filled, diff_range
    global dParam, zParam

    print("\n")
    print("Constants from the configuration file")
    print("L, L_filled, diff_range in main: ", L, L_filled, diff_range)
    print("dParam, zParam in main: ", dParam, zParam)

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
    iniLattice[0,0]=2 # for debugging

    return iniLattice

def findCellContributions(cellarray):
    """
    Finds number of contributors to each cell
    Input:  array of cells
    Output: array (same size) of contributor numbers for each cell
    """
    global L
    ### array of "2" coordinates
    coord_2_tuple = np.where(cellarray == 2)
    coord_2_array = np.asarray(coord_2_tuple).T
    print(coord_2_array)

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
    dists = [[np.linalg.norm(i-j) for i in coord_2_array] for j in coords]
    print("dists")
    print(dists)
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
    contribList = list(map(contribForCell,dists))
    #print(contribList)

    contribArray = np.array(contribList).reshape(L,L)
    return contribArray


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
    print("Finding cell benefits")

def main():
    ### initializations
    initializeGlobals()            # global vars from config file
    printGlobalVars()
    cellArray = iniSquareLattice() # lattice
    print(cellArray)

    ### calculate benefits for each cell
    contribArray = findCellContributions(cellArray) # returns square array of contributors
    print("contribution: ")
    print(contribArray.round(decimals=2))
    #benefits     = normBenefitFn(cellArray,contribution)

    ### play the game as described in the paper

    ## Benefits
    # p.3: weighted number of producers <- sum of all contributions
    # benefit f'n = f(sum of contribs/group size)
    benefitArray = findCellBenefits(contribArray)

    ## birth - death game
    


if __name__=="__main__":
    main()
