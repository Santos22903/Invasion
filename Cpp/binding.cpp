/*
  06/02/22 S.Tkachenko
  C++ version of the binding step called from doStepping
  
 */
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>

namespace py = pybind11;
using namespace std;

tuple<Eigen::MatrixXi,Eigen::Matrix<int,Dynamic,Dynamic>> bindingStep(Eigen::MatrixXi benefarray, Eigen::Matrix<int,Dynamic,Dynamic>gfcoords,
								      int lattice_size, bool bind_1reaction)
{
  /*
    Routine performing binding after each diffusion time step. 
    
    Input: benefarray     - how much "public good" already bound at each node
           gfcoords       - coordinates of free floating "goodies"
	   lattice_size   - global L in python - size of 1 square lattice dimension
	   bind_1reaction - whether to consider only 1 reaction possible in diff_delt
    Output: 1st 2 inputs after possible binding in the diff_delt time
            (time it takes to travel unit step - b/n nodes)
  */

    # converting to "lattice matrix"
    coordmat = coord_to_mat(gfcoords.to_numpy(dtype="int"))

    #print("Before gillespie: benefarray[3,3] ",benefarray[3,3]," coordmat[3,3] ",coordmat[3,3])
    #boundben, freeben = GillespieFn(benefBound=benefarray[3,3],benefFree=coordmat[3,3])
    #print("After gillespie: boundben ",boundben," freeben ",freeben)

    # bind_1reaction defines if only 1 possible reaction per binding
    # step is considered - yes/no, then exit; if false, then after
    # 1 reaction occured, probability of the next one before the step
    # time is over is checked - quite unlikely for parameters used on
    # 05/17/22 (vegf stuff) and slows down a lot

    benefarray_1d = benefarray.flatten(); coordmat_1d = coordmat.flatten()
    #start = timer()
    #rands0 = rng.random(size=L*L); rands1 = rng.random(size=L*L)
    #end = timer()
    #print("Time of generation: ",end-start)
    if bind_1reaction:
        rands0 = rng.random(size=L*L); rands1 = rng.random(size=L*L)
        if useMultiproc:
            pool = mp.Pool(cores)
            resarray = np.array(pool.starmap(GillespieFn_1react,zip(benefarray_1d,coordmat_1d,rands0,rands1)),dtype="int")
            pool.close()
        elif useMap:
            resarray = np.array(list(map(GillespieFn_1react,benefarray_1d,coordmat_1d,rands0,rands1)),dtype="int")
        else:
            resarray = np.array([GillespieFn_1react(x,y,z,a) for x,y,z,a in zip(benefarray_1d,coordmat_1d,rands0,rands1)],dtype="int")
    else:
        if useMultiproc: # b/c of RNG, had to code a separate routine
            pool = mp.Pool(cores)
            ss = rng.bit_generator._seed_seq
            child_states = ss.spawn(L*L)
            resarray = np.array(pool.starmap(GillespieFn_multi,zip(benefarray_1d,coordmat_1d,child_states)),dtype="int")
            pool.close()
        elif useMap:
            resarray = np.array(list(map(GillespieFn,benefarray_1d,coordmat_1d)),dtype="int")
        else:
            resarray = np.array([GillespieFn(x,y) for x,y in zip(benefarray_1d,coordmat_1d)],dtype="int")
    

    benefarray_1d = resarray[:,0]; benefarray = benefarray_1d.reshape(L,L)
    coordmat_1d = resarray[:,1];   coordmat   = coordmat_1d.reshape(L,L)

    # converting back to dataframe
    gfcoords = mat_to_coord(coordmat)

    return benefarray, gfcoords
