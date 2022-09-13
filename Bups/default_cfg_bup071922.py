P_mut=0.01
#D=1.59e-6 # cm^2/s for IGF1 through fibrin gel from nauman et al, 2007
Rprod0=50 # per cell per cycle time production rate of the GF
inflx=0.5
steep=0.1
n_cycles=10 # number of cell proliferation cycles to run for
L=15      # linear dimension of square lattice
L_filled=15 # filled part at center of lattice
cellcycle_t = 86400 # s, cell cycle time, 24 hrs in seconds
#secrete_tstep = 195 # s, time between gf vbrosy, <t> of 1 VEGF molecule secretion (adipocyte paper)
secrete_tstep = 1.95 # s, RECALCULATED IN JULY 2022 
secreteN = 1       # number of molecules excreted by each "2" cell when time comes. 10 FOR TESTING DIFFUSION, RETURN TO 1 AFTER DONE
cost       = 0.0# cost of producing benefits; initial=0.1 taken as mid of fig 2 range
useCost    = False # whether to account for cost of PG production
#diffus_tstep = 2.22e-2 # time step for diffusion
diff_coef = 1e6/3600 # um^2/h estimate for vegf from miura/tanaka converted to um^2/s, ~277
#diff_coef = 104 #gabhann/ji/popel
delx = 14.6          # um, calculated from tumor cell density from Lyng et al (3d cube was imagined for their density=>()^(1/3), inverted)
k_on = 3.6e6         # 1/(M*s), for VEGF165+VEGFR2, from Gabhann/Yang/Popel, their ref [23]
#k_on = 1e6           # VEGF165+VEGFR2 from mamet et al
k_off = 1.34e-4       # 1/s, for VEGF165+VEGFR2, from Gabhann/Yang/Popel, their ref [23]
receptors_per_cell = 25000 # from Gabhann/Popel
um_to_dm = 1e-5       # micrometer to decimeter conversion factor
EC50 = 7.76e-11       # M, VEGF165 - VEGFR2 EC50 from Whitaker-Limberg-Rosenbaum, 2001
n_hill = 1            # hill coefficient
## code flow constants
bind_1reaction = True  # if true, gillespie will consider 1 reaction and exit
useMultiproc   = False  # whether to use multiprocessing in calling GillespieFn
useMap         = True   # whether to use mapping or list comprehension if not multiproc

##
updateMethod = "pairwise"
invasion_done_perc = 0.5 # fraction of invaders to have invasion succeed
