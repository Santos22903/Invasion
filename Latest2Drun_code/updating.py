import numpy as np
import math

def strategyUpdate(cellarray,fitnesses,method):
    print("In strategy update")
    VALID_METHODS = {"pairwise","moran","wf"} # last standing for Wright-Fisher
    if method not in VALID_METHODS:
        raise ValueError("strategyUpdate: method must be one of %r." % VALID_METHODS)

    """ THE FOLLOWING WILL WAIT TILL PYTHON 3.10 IS STABLE/STANDARD
    match method:
    case "pairwise":
        return pairwiseUpdate()
    case "moran":
        return moranUpdate()
    case "wf":
        return writeFisherUpdate()
    case _:
        print("Illegal method")
        return 0
    """

    if method=="pairwise":
        return pairwiseUpdate(cellarray,fitnesses)
    elif method=="moran":
        return moranUpdate(cellarray,fitnesses)
    elif method == "wf":
        return writeFisherUpdate(cellarray,fitnesses)
    else:
        print("Illegal method")
        return 0


### 09/16/22 added 1's for the case when all fitnesses = 0
### 09/16/22 adjusted proportFactor so that would always get at least lattice
###          size number of cells
### 10/06/22 Added posibility of expanding lattice
def wrightFisherSpatial(cellarray,fitnesses,rng_arg,expanding,newL=None):
    """
    Updating a-la Wright-Fisher, but preserving spatial structure
    Classical part:
    Each cell produces a large number of offspring ~ fitness (bound benefits)
    Then N cells are randomly sampled from the resulting pool
    Spatial structure part:
    all the "2" (if present) set next to each other in the center
    """
    #print("Write-Fisher update preserving spatial structure")
    #print("expanding ",expanding)
    # checking that newL is not None if expanding
    if expanding:
        assert newL is not None
        
    # each cell producing offspring
    proportFactor = 1000
    # number of cells on the lattice
    cellarray_1d = cellarray.flatten()
    if expanding:
        cellNumber = newL * newL
    else:
        cellNumber = len(cellarray_1d)
    ### 09/16/22 if all fitnesses are 0, add ones for workflow to continue
    fitn_sum = np.sum(fitnesses)
    if fitn_sum==0:
        fitnesses = fitnesses + 1
        fitn_sum = np.sum(fitnesses) # also readjust fitnessess sum for future use
    ### 09/16/22 another problem: if fitnesses are !0's, but few goit smth
    ### and when multiplied by proportFactor, do not get enough cells
    if fitn_sum*proportFactor < cellNumber:
        proportFactor = math.ceil(cellNumber/fitn_sum)
        #print("cellNumber ",cellNumber)
        #print("fitn_sum ",fitn_sum)
        #print("new proportFactor ",proportFactor)
        #print("np.ceil(fitnesses * proportFactor) ",np.ceil(fitnesses * proportFactor))
    
    #offspring_nums = np.round(fitnesses * proportFactor).astype(int)
    offspring_nums = np.ceil(fitnesses * proportFactor).astype(int)
    offspring_array = np.repeat(cellarray_1d,offspring_nums)
    #print("offspring_nums ",offspring_nums)
    #print("offspring_array")
    #print(offspring_array)
    #print("len(offspring_array) ",len(offspring_array))

    ### sampling the result
    final_array = rng_arg.choice(offspring_array,size=cellNumber,replace=False)
    #print("final_array")
    #print(final_array)
    #print("2s: ",np.where(final_array==2))
    
    ### if "2" are present, cluster in the middle
    # 0) put "1" everywhere ("initialization" - neglecting "0" for the time being)
    # 1) make 1d array of size (ceil(sqrt(N)))^2 <- "padded" to full square, reshape
    #    to ((ceil(sqrt(N)),(ceil(sqrt(N)))
    # 2) assign resulting 2d array to be part of big cellarray as in iniSquareLattice(),
    #    just having this array instead of repeating "1"

    ## 0)
    if expanding:
        side = newL
    else:
        side = len(cellarray)
    new_cellarray = np.ones((side,side))

    ## 1)
    N = np.count_nonzero(final_array==2) # number of "2"'s
    # if the resulting number of "2"s is larger than fits in lattice, return
    # lattice filled with all "2" and give warning
    if N>=cellNumber:
        print("Number of 2's exceeded lattice size, returning lattice full of 2's")
        new_cellarray = np.full((side,side),2)
        return new_cellarray
    #print("Number of 2's ",N)
    side_of_subarray = math.ceil(math.sqrt(N))
    array_of_2_1d = np.ones(side_of_subarray*side_of_subarray) # "initialize" to all "1"
    array_of_2_1d[0:N] = 2                                 # stick proper # of "2"'s in there
    #print("array_of_2_1d")
    #print(array_of_2_1d)
    # reshape to 2d and insert into the "initial" array of "1"'s
    array_of_2_2d = array_of_2_1d.reshape(side_of_subarray,side_of_subarray)
    loFilled = (side-side_of_subarray)//2
    if loFilled<0: # in case below lowest index
        loFilled = loFilled + 1
    hiFilled = loFilled + side_of_subarray #?unnecessary to catch hiFilled>L, b/c loFilled condition + checking for exceeding lattice size above should account for this?
    new_cellarray[loFilled:hiFilled,loFilled:hiFilled] = array_of_2_2d

    return new_cellarray
    
    
def pairwiseUpdate(cellarray,fitnesses):
    print("pairwise update")
    # randomly choose 2 cells

    # get probability of focal accepting "role model" strategy (?linear - traulsen eq 1.5)

    # seeding probability - accepting strategy or not

    # changing strategy if yes (then benefits of focal -> 0)

def moranUpdate(cellarray,fitnesses):
    print("Moran update")
    # choose individual ~ fitness

    # randomly choose the one to die

    # assign chosen individual

def wrightFisherUpdate(cellarray,fitnesses):
    print("Wright-Fisher update")
