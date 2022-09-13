"""
Benefit functions (might use different ones) collected here
"""

def hillFn(n_bound, EC50_bound=10, n=1):
    """
    INPUT
    EC50_bound provided is ligand concetration in "number/node" converted to "bound"
               by finding rate_on as in Gabhann/Popel and * by cell cycle time)
    n    hill coefficient

    OUTPUT
    "benefit" of a cell at the end of a cell cycle
    """

    # no goodies bound = no benefits
    if n_bound==0:
        return 0
    
    # take bound to be same as EC50 - given enough time and receptors, they'll bind
    ratio = EC50_bound/n_bound

    ## calculating benefit
    benefit = 1/(1+pow(ratio,n))

    return benefit
