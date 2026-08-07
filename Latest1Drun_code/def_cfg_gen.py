#P_mut=0.01
#D=1.59e-6 # cm^2/s for IGF1 through fibrin gel from nauman et al, 2007
#Rprod0=50 # per cell per cycle time production rate of the GF
#inflx=0.5
#steep=0.1
n_cycles=10 # number of cell proliferation cycles to run for: changed 10->25, 09/20/22
L=100      # linear dimension of square lattice - initial value in "expanding", +1 each cycle
L_filled=100 # initially filled part at center of lattice - initial value in "expanding", +1 each cycle
cellcycle_t = 86400 # s, cell cycle time, 24 hrs in seconds
#secrete_tstep = 195 # s, time between gf vbrosy, <t> of 1 VEGF molecule secretion (adipocyte paper)
secrete_one_tstep = 1.95 # s, RECALCULATED IN JULY 2022 
secreteN = 1000       # number of molecules excreted by each "2" cell when time comes. 10 FOR TESTING DIFFUSION, RETURN TO 1 AFTER DONE
#cost       = 0.1  # cost of producing benefits; initial=0.1 taken as mid of fig 2 (NEED PAPER!?!) range
useCost    = False # whether to account for cost of PG production
#diffus_tstep = 2.22e-2 # time step for diffusion
#diff_coef = 1e6/3600 # um^2/h estimate for vegf from miura/tanaka converted to um^2/s, ~277
### diffusion coefficients for parameter sweep, Aug 30, 2022, see notes, expressed in um^2/s
#diff_coef = 3
#diff_coef = 47
#diff_coef = 90
#diff_coef = 134
#diff_coef = 178
#diff_coef = 222
#diff_coef = 265
#diff_coef = 309
#diff_coef = 353
#diff_coef = 396
#diff_coef = 440
#########################
#diff_coef = 104 #gabhann/ji/popel
delx = 14.6          # um, calculated from tumor cell density from Lyng et al (3d cube was imagined for their density=>()^(1/3), inverted)
#k_on = 3.6e6         # 1/(M*s), for VEGF165+VEGFR2, from Gabhann/Yang/Popel, their ref [23]
######################### k_on values for the sweep
#k_on = 1e3
#k_on = 1e4
#k_on = 1e5
#k_on = 1e6
#k_on = 1e7
#k_on = 1e8
#########################
#k_off = 1.34e-4       # 1/s, for VEGF165+VEGFR2, from Gabhann/Yang/Popel, their ref [23]
######################### k_off values for the sweep
#k_off = 1e-1
#k_off = 1e-2
#k_off = 1e-3
#k_off = 1e-4
#k_off = 1e-5
#k_off = 1e-6
#########################
receptors_per_cell = 25000 # from Gabhann/Popel
um_to_dm = 1e-5       # micrometer to decimeter conversion factor
EC50 = 7.76e-11       # M, VEGF165 - VEGFR2 EC50 from Whitaker-Limberg-Rosenbaum, 2001
n_hill = 1            # hill coefficient
## code flow constants
bind_1reaction = False  # if true, gillespie will consider 1 reaction and exit
useMultiproc   = False  # whether to use multiprocessing in calling GillespieFn
useMap         = True  # whether to use mapping or list comprehension if not multiproc
#
n_runs    = 50           # how many independent runs to perform
cores2use = 16           # multiprocessing cores

##
invasion_done_perc = 0.05 # fraction of invaders to have invasion succeed

##
runs_in_parallel = True

## benefit offset from 0 in units of # of bound PG (-> benefits via hillFn)
bound_PG_offset = 0
bound_benef_offset = 0
use_bound_benef_offset = False

## determining which (if any) diagnostic output to make
benef_frac_out = False
cycleN_out     = False
boundSum_out   = False
GF_sums_list   = False
gf_spat_profile= False
benef_size_out = False

## fixed or expanding (and how) size
fixedSize = True # whether lattice size is fixed
expandDet = False # whether expansion is deterministic (+1 each prolif cycle) or ~fitness; dummy if fixedSize 

## output to screen or file
out2screen = False

## debugging flags
DIFFUSION_DEBUG = False
BINDING_DEBUG   = False
RNG_DEBUG       = False
CC_DEBUG        = False
