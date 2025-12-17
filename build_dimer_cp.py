# -----------------------------------------------------------------#
# Builds supercells of close packed hard sphere dimer structures. 
#
# Dimers consist of two hard spheres of radius sigma, connected by 
# a rigid bond of length L. Structures should be valid for L <= sigma. 
#
# Structures consist of close packed layers of spheres. The first 
# layer contains spheres each constituting one end of a dimer. The 
# second close packed layer constitutes the other end of these dimers
# such that the dimers are tilted at arcsin[L/(sigma*sqrt(3))]. As 
# argued by Vega, Paras and Monson, this leads to maximum packing 
# of spheres into these two layers (one bilayer).
#
# We refer to the vector from the centre of a sphere in the first 
# layer to the centre of its partner in the next layer as the 
# dimer shift vector. 
# 
# The next bilayer is shifted from the first such that the lower 
# layer of close packed sphere sits in the hollow sites formed 
# by the upper layer of the previous bilayer. i.e. the vector 
# between the upper bead of bilayer n and the lower bead of bilayer
# n+1 is the standard close packing layer shift vector.

# Note that for L=1 the dimer shift vector and layer shift vector 
# are of equal magnitude and angle to the vertical (z) direction, 
# recovering hard sphere (fcc, hcp etc) packing of the constituent
# spheres. 
# 
# If the in-plane component of the layer shift and dimer shift are 
# are parallel and consistent across all layers, the structure is
# the CP1 structure of Vega, Paras and Monson. This reduces to
# fcc (ABC)n stacking of the constituent spheres for L=1.
# 
# If the in-plane component of the layer shift and dimer shift are
# anti-parallel and consistent across all layers, the structure is 
# the CP2 structure of Vega, Paras and Monson. This reduces to 
# HCP stacking (AB)n of the constituent spheres for L=1.
# 
# If the in-plane component of the dimer shift vector alternates
# parallel to antiparallel between layers, but the layer shift
# remains constant, a third structure is produced. This has 
# (ACABCBCABABC)n stacking of the constituent spheres for L=1. 
#
# If both in-plane components of the dimer shift vector and the layer
# layer shift vector alternate directions, but remain parallel to 
# each other, a further structures is produced. This has 
# (ACAB)n (double HCP) stacking of the constituent spheres for L=1. 
#
# It is not clear which of the last two structures is that which has 
# been referred to as CP3 by Vega, Paras and Monson.
#
# Many other layer and dimer shift sequences are possible, although
# a particular combination of L and shift sequence may not lead to 
# a structure that can be represented with a unit cell of tractable
# size.
#
# An additional degree of freedom can be introduced by having one sphere
# in each unit cell of a close packed layer being bonded to the layer above,
# and one to the layer below. The dimers are hence interdigitated between
# layers. This leads to a different set of layer and dimer shift vectors.
# The possibilities are endless.....
#
# This version only allows for the dimer and layer shift vectors to 
# lie in a single plane. Valid structures exist in which these
# vectors are non-coplanar and rotate pi/3 between layers.
#        
# D. Quigley May 2025 
# -----------------------------------------------------------------#

#import hoomd
import numpy as np
import math as m

# Use ase For building crystal structures
from ase import Atoms
from ase.build import make_supercell

# Use HOOMD-blue style gsd file for input/output
import gsd.hoomd

# DQ extra functions for manipulating rigid bodies
import dqutils as utils

# ------------------#
# Utility functions #
# ------------------#
def closest_multiple(n, m):
    """ Calculate the nearest multiple of n to m. Written by copilot """

    # Calculate the quotient
    quotient = m // n

    # Calculate the two closest multiples
    lower_multiple = n * quotient
    upper_multiple = n * (quotient + 1)

    # Determine which multiple is closer to m
    if abs(m - lower_multiple) < abs(m - upper_multiple):
        return lower_multiple
    else:
        return upper_multiple
    
# ---------------------------------------------------------------------#
# Functions for computing geometry of cell in interdigitated case      #
# ---------------------------------------------------------------------#
def dimer_shift_interdig(sigma, L):
    '''Shift vector between the two ends of a dimer in interdigitated structure'''

    dimer_shift = []

    lstar = L/sigma

    delta = m.sqrt(3-lstar**2)
    alpha = 6.0-lstar**2+2*m.sqrt(2)*lstar*delta

    denom = m.sqrt(3*alpha)
    
    xshift = 0.0
    yshift = L*(lstar+m.sqrt(2)*delta)/denom
    zshift = L*(m.sqrt(2)*lstar+2*delta)/denom

    dimer_shift.append(np.array([xshift, -yshift, zshift]))
    dimer_shift.append(np.array([xshift, +yshift, zshift]))

    return dimer_shift

def layer_shift_interdig(sigma, L):
    '''Shift vector from upper sphere in one layer to lower sphere in next layer in interdigitated structure'''
    layer_shift = []

    lstar = L/sigma

    delta = m.sqrt(3-lstar**2)
    alpha = 6.0-lstar**2+2*m.sqrt(2)*lstar*delta

    ydenom = m.sqrt(3)*m.sqrt(alpha)
    zdenom = alpha*ydenom

    xshift = 0.0
    yshift = L*(lstar+m.sqrt(2)*delta)/ydenom
    zshift = sigma*( m.sqrt(2)*(18 - 3*lstar**2 - lstar**4) + (18*lstar-5*lstar**3)*delta)/zdenom

    layer_shift.append(np.array([xshift, -yshift, zshift]))
    layer_shift.append(np.array([xshift, +yshift, zshift]))

    return layer_shift

def unit_cell_interdig(sigma, L):
    ''' Returns the a, b and c lattice parameters for a unit cell of one layer in an interdigitated structure '''
    
    lstar = L/sigma  

    c1 = sigma

    c2 = sigma*m.sqrt(6-lstar**2+2*lstar*m.sqrt(6-2*lstar**2))/m.sqrt(3.0)

    c3n = 2*m.sqrt(3)*(-6*m.sqrt(2)-5*m.sqrt(2)*lstar**2+2*m.sqrt(2)*lstar**4-10*lstar*m.sqrt(3-lstar**2)+(lstar**3)*m.sqrt(3-lstar**2))
    c3d = m.sqrt((6-lstar**2+2*lstar*m.sqrt(6-2*lstar**2))**3)
    c3 = sigma*c3n/c3d

    return [c1, c2, c3]

def basis_shift_interdig(sigma, L):
    ''' Returns the shift vector between the two dimers in the unit cell of a single layer in an interdigitated structure '''

    lstar = L/sigma  

    delta = m.sqrt(3-lstar**2)
    alpha = 6.0-lstar**2+2*m.sqrt(2)*lstar*delta

    xshift = 0.5*sigma

    ydenom = 2*m.sqrt(alpha)
    zdenom = ydenom*alpha

    yshift = sigma*(m.sqrt(3)*(2-lstar**2))/ydenom
    zshift = sigma*m.sqrt(3)*(m.sqrt(2)*(6+5*lstar**2-2*lstar**4) + (10*lstar - lstar**3)*delta)/zdenom

    return [xshift, yshift, zshift]

# ----------------#
# Parse arguments #
# ----------------#
import argparse
parser=argparse.ArgumentParser(description="Builds close packed structures of dimers and writes to lattice.gsd")

parser.add_argument("sigma", type=float, help="Diameter of beads on dimer")
parser.add_argument("L", type=float, help="Dimer bond length")
parser.add_argument("Na", type=int, help="Number of unit cells in a direction")
parser.add_argument("Nb", type=int, help="Number of unit cells in b direction")
parser.add_argument("Nl", type=int, help="Number of stacked hexagonal layers")
parser.add_argument("--dimer_shifts", default="constant", choices=["constant", "alternating", "random"], help="Bond shift vector")
parser.add_argument("--layer_shifts", default="constant", choices=["constant", "alternating", "random"], help="Layer shift vector")
parser.add_argument("--initial_shifts", default="equal", choices=["aligned", "opposite"], help="Bond and layer shifts initially aligned/opposite?")
parser.add_argument("--interdig", action=argparse.BooleanOptionalAction, default=False, help="Interdigitate bonds between layers")


args=parser.parse_args()

# --------------------------------- #
# Define dimers, angles and shifts  #
# --------------------------------- #

# prototype dimer - aligned parallel to z axis
prototype = utils.DimerPrototype(args.sigma, args.L)

# For maximum packing in a layer of dimers, we have a hexagonal unit cell
# in which each dimer is tilted to the normal of that layer by...
tilt_sin = args.L/(args.sigma*m.sqrt(3.0))
tilt_cos = m.sqrt(1.0 - tilt_sin**2)

tilt_sinL1 = args.sigma/(args.sigma*m.sqrt(3.0))
tilt_cosL1 = m.sqrt(1.0 - tilt_sinL1**2)

# Dimensions of the unit cell for that hexagonal base layer.
if (args.interdig):

    a_lat = unit_cell_interdig(args.sigma, args.L)[0]
    b_lat = unit_cell_interdig(args.sigma, args.L)[1]

    dimer_shift = dimer_shift_interdig(args.sigma, args.L)
    layer_shift = layer_shift_interdig(args.sigma, args.L)

else:

    a_lat = args.sigma
    b_lat = 2.0*a_lat*m.cos(m.pi/6.0)

    # The next layer of dimers sits in the hollow sites created by the upper beads of
    # the first layer. So from upper bead of one dimer layer to lower bead of the next
    # will be one of...
    layer_shift = []
    layer_shift.append(np.array([0.0, -args.sigma/m.sqrt(3.0), 0.5*args.sigma*m.sqrt(8.0/3.0)]))
    layer_shift.append(np.array([0.0, +args.sigma/m.sqrt(3.0), 0.5*args.sigma*m.sqrt(8.0/3.0)]))


    # Shift in coordinates from one end of a dimer to the other. Two possibilities if we allow for 
    # changes in dimer shift from one layer to the next as in the CP3 structure.
    dimer_shift = []
    dimer_shift.append(np.array([0.0, -args.L*tilt_sin, args.L*tilt_cos]))
    dimer_shift.append(np.array([0.0, +args.L*tilt_sin, args.L*tilt_cos]))


# ----------------------------------------------------------------------------- #
# Construct shift sequences and identify number of layers for periodic stacking #
# ----------------------------------------------------------------------------- #
# Figure out how many layers we need for the y coordinate in the last plus one
# layer to be the same as the first layer.

# List of layer and dimer shift types
layer_shifts_sequence = []
dimer_shifts_sequence = []

# First step in y direction - first layer and dimer shift always have parallel y components

dtype = 0 # first dimer shift always zero
dtype0 = dtype

ltype = 0 # default ltype in case argument not supplied

# first layer shift might be in same or opposite direction, or random
if args.layer_shifts == 'constant' or args.layer_shifts == 'alternating':
    
    if args.initial_shifts == 'aligned':
        ltype = 0
    elif args.initial_shifts == 'opposite':
        ltype = 1

elif args.layer_shifts == 'random':
    
    ltype = np.random.randint(0,2)  

ltype0 = ltype # Store first dimer shift

layer_shifts_sequence.append(ltype)
dimer_shifts_sequence.append(dtype)

y_step = layer_shift[ltype][1] + dimer_shift[dtype][1]

tol = 0.01

# If the step size is zero we're done - stacking periodicity in a single bilayer
if (abs(y_step) < tol and args.dimer_shifts=='constant' and args.layer_shifts=='constant'):
    n_layers=1
else:

    span = y_step

    maxtry = 100 # Bail out if we don't find a periodic stacking after 100 bilayers
    for i in range(1,maxtry+1):

        if args.layer_shifts == 'constant':
            ltype = ltype0
        elif args.layer_shifts == 'alternating':
            ltype = (i+ltype0)%2
        elif args.layer_shifts == 'random':
            ltype = np.random.randint(0,2)

        layer_shifts_sequence.append(ltype)

        if args.dimer_shifts == 'constant':
            dtype = dtype0
        elif args.dimer_shifts == 'alternating':
            dtype = (i)%2
        elif args.dimer_shifts == 'random':
            dtype = np.random.randint(0,2)  

        dimer_shifts_sequence.append(dtype)  

        y_step = layer_shift[ltype][1] + dimer_shift[dtype][1]

        span += y_step

        if span > b_lat:
            span = span - b_lat

        if span < 0.0:
            span = span + b_lat
        
        if (abs(span) < tol) or (abs(b_lat - span) < tol or (abs(b_lat + span < tol))) :

            # If we have an alternating layer sequence, only exit if i+1 is even
            if args.layer_shifts=="alternating" or args.dimer_shifts=="alternating":
                if (i+1)%2==0:
                    n_layers = i+1
                    break
            else:
                n_layers = i+1
                break

        if (i==maxtry):
            print("No suitable unit cell size found, reduce tol or maxtry")
            exit()


print("Minimum number of layers for z periodicity : ", n_layers)

if (args.Nl < n_layers):
    Nl = n_layers
    print("Using minimum number of ", Nl, "layers.")
else:
    Nl = closest_multiple(n_layers, args.Nl)
    print("Using ", Nl, "layers as closest multiple of ", n_layers, "to ", args.Nl)

dimer_shifts_sequence = dimer_shifts_sequence*(Nl//n_layers)
layer_shifts_sequence = layer_shifts_sequence*(Nl//n_layers)

c_lat = ( layer_shift[0][2] + dimer_shift[0][2] ) * Nl

# Print sequence
dshift_labels = ['r', 'l']
lshift_labels = ['R', 'L']

shiftstring = []
for i in range(Nl):
    shiftstring.append(dshift_labels[dimer_shifts_sequence[i]])
    shiftstring.append(lshift_labels[layer_shifts_sequence[i]])
print("Shift sequence: " +"".join(shiftstring))

shiftstring = []
for i in range(Nl):
    shiftstring.append(dshift_labels[(dimer_shifts_sequence[i]+1)%2])
    shiftstring.append(lshift_labels[(layer_shifts_sequence[i]+1)%2])
print("Shift sequence: " +"".join(shiftstring))

if args.L == 1.0:

    abc = ["A", "B", "C"]

    layer_string = []
    layer_pos = 0
    layer_string.append(abc[layer_pos%3])

    drn = [+1, -1]

    for i in range(Nl-1):
        layer_pos = layer_pos + drn[dimer_shifts_sequence[i]]
        layer_string.append(abc[layer_pos%3])
        layer_pos = layer_pos + drn[layer_shifts_sequence[i]]
        layer_string.append(abc[layer_pos%3])

    print("Layer sequence: " +"".join(layer_string))

###################
# Build unit cell #
###################

# 2 dimers per layer in conventional unit cell
mcells = args.Na*args.Nb*Nl
N_dimers = 2 * mcells  # Total number of dimers (NOT beads)


print()
print("Creating supercell with ",N_dimers," dimers")
print("===========================================")

# The "position" of the dimer should be stored as the position of
# its zeroth bead. 

# Set orientation consistent with tilt angle
coshalf = m.sqrt(0.5*(1.0 + tilt_cos))
sinhalf = m.sqrt(0.5*(1.0 - tilt_cos))

coshalfL1 = m.sqrt(0.5*(1.0 + tilt_cosL1))
sinhalfL1 = m.sqrt(0.5*(1.0 - tilt_cosL1))

positions = list()
orientation = list()

pos1 = np.array([0.001, 0.001, 0.001])

if args.interdig:
    pos2 = pos1 + np.array(basis_shift_interdig(args.sigma, args.L))

else:
    pos2 = np.array([0.001+0.5*a_lat, 0.001+0.5*b_lat, 0.001])


sign = [1.0, -1.0]


for il in range(Nl):

    dtype = dimer_shifts_sequence[il]
    ltype = layer_shifts_sequence[il]

    if args.interdig:

        # First dimer
        pos = pos1
        if (pos[1] > b_lat): pos[1] -= b_lat
        if (pos[1] < 0.0  ): pos[1] += b_lat
        positions.append(pos)

        pos = pos2
        if (pos[1] > b_lat): pos[1] -= b_lat
        if (pos[1] < 0.0  ): pos[1] += b_lat
        positions.append(pos)

        # Advance one layer shift and one dimer shift for next dimer position
        pos1 = pos1 + dimer_shift[dtype] + layer_shift[ltype]
        orientation.append(np.array([sign[dtype]*coshalfL1, sinhalfL1, 0, 0]))

        lntype = layer_shifts_sequence[(il+1)%Nl] # next layer shift type
        dntype = dimer_shifts_sequence[(il+1)%Nl] # next dimer shift type
        pos2 = pos2 + dimer_shift[ltype] + layer_shift[dntype]
        orientation.append(np.array([sign[ltype]*coshalfL1, sinhalfL1, 0, 0]))

    else:


        # First dimer
        pos = pos1
        if (pos[1] > b_lat): pos[1] -= b_lat
        if (pos[1] < 0.0  ): pos[1] += b_lat
        positions.append(pos)

        # Second dimer
        pos = pos2
        if (pos[1] > b_lat): pos[1] -= b_lat
        if (pos[1] < 0.0  ): pos[1] += b_lat
        positions.append(pos)

        # Advance on layer shift and one dimer shift for next dimer position
        pos1 = pos1 + dimer_shift[dtype] + layer_shift[ltype]
        orientation.append(np.array([sign[dtype]*coshalf, sinhalf, 0, 0]))

        pos2 = pos2 + dimer_shift[dtype] + layer_shift[ltype]
        orientation.append(np.array([sign[dtype]*coshalf, sinhalf, 0, 0]))


symbols = f"C{Nl*2}"

# Atom positions and cell as a ASE atom object
# stretched slightly to make sure there's no overlaps
scale = 1.01
atoms = Atoms(symbols=symbols,
              positions = scale*np.array(positions),

              cell=[[scale*a_lat,0,0],
                    [0,scale*b_lat,0],
                    [0,0,scale*c_lat]],

              pbc=True)

P = np.array([ [args.Na,0.0,0.0] , [0.0,args.Nb,0.0] , [0.0,0.0,1] ] )
supercell = make_supercell(atoms, P, order="cell-major")

# Bash into format HOOMD likes, noting that HOOMD has the origin
# at the centre of the simulation box, not the corner
celllengths = supercell.cell.lengths()
position = [tuple(row-0.5*celllengths) for row in supercell.positions]
orientation = orientation *args.Na*args.Nb

# Box is specified as per LAMMPS, with 3 length and 3 tilt factors
initial_box =  celllengths.tolist() + [0, 0, 0]
print("Cell lengths :", celllengths.tolist())

# Check for overlaps before getting too excited - this is slow for large N so commented out by default
#print("Checking for overlaps in generated configuration")
#hmatrix = supercell.cell
#overlaps = utils.check_sphere_overlaps(position, orientation, prototype, hmatrix)
#if (len(overlaps) > 0):
#    print("Overlaps in initial configuration!")
#    print(overlaps)

# -------------------------------- #
# Write as a HOOMD to lattice.gsd  #
# -------------------------------- #
frame = gsd.hoomd.Frame()
frame.particles.N = N_dimers
frame.particles.typeid = [0] * N_dimers

# Set positions and orientations as created above
frame.particles.position = position
frame.particles.orientation = orientation
frame.configuration.box = initial_box

with gsd.hoomd.open(name='lattice.gsd', mode='w') as f:
    f.append(frame)

print("CP dimer structure written to lattice.gsd for HOOMD-blue")

# --------------------------------------- #
# Create extra formats for visualisation  #
# --------------------------------------- #
hmatrix = supercell.cell
utils.hoom2xyz(position, orientation, prototype.positions, hmatrix, 'lattice.xyz')
print("xyz file (for dinosaurs) written to lattice.xyz")
print("")

print("Dimers per unit volume : ",N_dimers/supercell.cell.volume)
print("Occupied vol fraction  : ",N_dimers*prototype.volume/supercell.cell.volume)
print("")


# utils.write_psf(position, prototype.positions)
# print("bond/topology file written to chain.psf (for VMD)")
# utils.write_dcd_header(position, prototype.positions)
# utils.write_dcd_snapshot(position, orientation, prototype.positions, hmatrix)
# print("DCD file (for VMD) written to chain.dcd")