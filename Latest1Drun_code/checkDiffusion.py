import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

DIFFUSION_DEBUG = False

def checkDiffusion1d():
    """
    09/29/23 
    repeating checkDiffusion3d for 1 dimension
    """

    steps2plot = [3, 10, 20, 50, 100, 1000]

    Nsteps_max = 1001
    # dictionary to fill w coords for appropr steps (keys) to later plot
    stepHistDictX = dict(zip(steps2plot,([] for _ in steps2plot)))

    # globals needed by imported routines
    global L, L_filled, secreteN
    L=15; L_filled=15; secreteN=1000

    # produce linear lattice
    cellArray = iniLinearLattice()

    # produce N GFs at the origin
    gfcoords = secreteGF(cellArray,oldGfDf=pd.DataFrame(columns=['x']))

    # do stepping, store coordinates at given step #s
    local_rng = np.random.default_rng(12345)
    for stepN in range(Nsteps_max):
        gfcoords = RWstep_v2(local_rng,gfcoords)
        if len(gfcoords)==0:
            print(f"step number {stepN}")
            break
        if stepN in steps2plot:
            # fill lists to later do hists - will not do hist() right
            # away to keep poss'ty to run loop over secreteGF, etc for
            # better statistics
            print(f"step number {stepN}")
            print(gfcoords)
            stepHistDictX[stepN].append(gfcoords.x)

    plotDiffusion1d(stepHistDictX,sidelength=L)
    
    return stepHistDictX

def plotDiffusion1d(stepHistDictX,sidelength):
    """
    10/02/23 Plotting part of checkDiffusion1d() moved here to also
    use in the main simulation
    """
    steps2plot = stepHistDictX.keys()
    plotColors = ["red","blue","green","black","silver","violet","cyan"]
    colDict = dict(zip(steps2plot,plotColors))
    # plot coordinates (histogram distributions saved in stepHistDict's)
    # overlay diff steps w diff color
    fig, ax_x = plt.subplots(1,1)
    fig.suptitle("Histograms of GF coordinates")
    for step in steps2plot:
        print(f"For step {step}")
        ax_x.hist(stepHistDictX[step],bins=sidelength,range=(0,sidelength),color=colDict[step],
                  label="Step "+str(step))
    fig.legend()
    print("Saving difftest_counts.pdf")
    plt.savefig("difftest_counts.pdf")
    #
    fig2, ax_x2 = plt.subplots(1,1)
    fig2.suptitle("Histograms of GF coordinate densities")
    for step in steps2plot:
        print(f"For step {step}")
        ax_x2.hist(stepHistDictX[step],bins=sidelength,range=(0,sidelength),color=colDict[step],
                   label="Step "+str(step), density=True)
       
    fig2.legend()
    print("Saving difftest_densities.pdf")
    plt.savefig("difftest_densities.pdf")
    
def checkDiffusion3d():
    """
    Routine checking diffusion by random walk function from invasion3D
    routines. For GF generation, lattice initialization also routines
    from invasion3D used. Plot coordinate distribution in x-y and x-z
    for given step #
    """
    
    steps2plot = [3, 10, 20, 50, 100, 1000]
    plotColors = ["red","blue","green","black","silver","violet","cyan"]
    colDict = dict(zip(steps2plot,plotColors))

    Nsteps_max = 1001
    # dictionaries to fill w coords for appropr steps (keys) to later plot
    stepHistDictX = dict(zip(steps2plot,([] for _ in steps2plot)))
    stepHistDictY = dict(zip(steps2plot,([] for _ in steps2plot)))
    stepHistDictZ = dict(zip(steps2plot,([] for _ in steps2plot)))
    
    # globals needed by imported routines
    global L, L_filled, secreteN
    L=15; L_filled=15; secreteN=1000

    # produce cubic lattice
    cellArray = iniCubicLattice()

    # produce N GFs at the origin
    gfcoords = secreteGF(cellArray,oldGfDf=pd.DataFrame(columns=['x','y','z']))
    
    # do stepping, store coordinates at given step #s
    local_rng = np.random.default_rng(12345)
    for stepN in range(Nsteps_max):
        gfcoords = RWstep_v2(local_rng,gfcoords)
        if len(gfcoords)==0:
            print(f"step number {stepN}")
            break
        if stepN in steps2plot:
            # fill lists to later do hists - will not do hist() right
            # away to keep poss'ty to run loop over secreteGF, etc for
            # better statistics
            print(f"step number {stepN}")
            print(gfcoords)
            stepHistDictX[stepN].append(gfcoords.x)
            stepHistDictY[stepN].append(gfcoords.y)
            stepHistDictZ[stepN].append(gfcoords.z)

    # plot coordinates (histogram distributions saved in stepHistDict's)
    # make three x-y-z panels, overlay diff steps in each w diff color
    fig, (ax_x, ax_y, ax_z) = plt.subplots(1,3)
    fig.suptitle("Histograms of GF coordinates")
    for step in steps2plot:
        print(f"For step {step}")
        ax_x.hist(stepHistDictX[step],bins=15,range=(0,L),color=colDict[step],
                  label="Step "+str(step))
        ax_y.hist(stepHistDictY[step],bins=15,range=(0,L),color=colDict[step])
        ax_z.hist(stepHistDictZ[step],bins=15,range=(0,L),color=colDict[step])
    fig.legend()    
    plt.savefig("difftest_counts.pdf")
    #
    fig2, (ax_x2, ax_y2, ax_z2) = plt.subplots(1,3)
    fig2.suptitle("Histograms of GF coordinate densities")
    for step in steps2plot:
        print(f"For step {step}")
        ax_x2.hist(stepHistDictX[step],bins=15,range=(0,L),color=colDict[step],
                   label="Step "+str(step), density=True)
        ax_y2.hist(stepHistDictY[step],bins=15,range=(0,L),color=colDict[step],
                   density=True)
        ax_z2.hist(stepHistDictZ[step],bins=15,range=(0,L),color=colDict[step],
                   density=True)
    fig2.legend()    
    plt.savefig("difftest_densities.pdf")
    
    return stepHistDictX

######## COPIED OVER
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
    gfcoords = gfcoords[~(gfcoords>=L).any(1)]
    gfcoords = gfcoords[~(gfcoords<0).any(1)]

    return(gfcoords)

######## COPIED OVER
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

######## COPIED OVER
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
