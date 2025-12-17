# ---------------------------------------------------#
# Tools to manipulate HOOMD-style rigid body objects #
# stored as positions and quaternions. Includes      #
# write to xyz and dcd given a prototype rigid body  #
# for which the quaternion is (1,0,0,0).             #
# ---------------------------------------------------#
import numpy as np
import math as m
#import dcd_tools as dcd  # for writing DCD files

class DimerPrototype():
    ''' Defines a reference dimer with CoM at the origin and aligned
        in the z direction. Positions and orientation of dimers are stored
        relative to this prototype. 
    '''
    
    def __init__(self, sigma, L):
        ''' Constructor. Creates prototype position and calculations volume 
    
        arguments:
            sigma           : sphere diameter
            L               : dimer bond length
        '''

        self.L = L           # Bond length     
        self.sigma = sigma   # Sphere radius

        # Positions of the two spheres in the dimer
        self.positions = [(0, 0, 0), (0, 0, L)]
        
        # Volume of the two spheres
        if (L > sigma):
            print("Warning - spheres on dimer do not overlap")
        else:
            h = 0.5*(L+sigma) ; r = 0.5*sigma
            self.volume = 2.0 * m.pi * (h**2) * (3*r-h) / 3.0

def quat_product(a, b, normalise=False):
    ''' Computes the product of two quaternions a and b '''

    tmpquat = np.zeros(4)
    tmpquat[0] = a[0]*b[0] - np.dot(a[1:],b[1:])
    tmpquat[1:] = a[0]*b[1:] + b[0]*a[1:] + np.cross(a[1:],b[1:])

    if (normalise):
        tmpquat = tmpquat/np.linalg.norm(tmpquat)

    return tmpquat

def quat_inverse(a):
    return np.array([a[0], -a[1], -a[2], -a[3]]) / np.dot(a, a)

def quat_conjugate_q_with_v(a, v):
    ''' Computes the conjugation of the quaternion q 
        with the vector v '''

    q = np.zeros(4)
    q[1:] = v[0:]

    b = quat_inverse(a)
    tmpquat = quat_product(q,b)
    q = quat_product(a,tmpquat)

    return q[1:]

def hoom2xyz(position, orientation, prototype, box, filename):
    ''' Writes an xmol stype xyz file listing the positions of all points
        which are part of rigid bodies in a hard particle simulation. The 
        comment line in the file is used to write the simulation cell vectors.

        paramters:
            positions   : rigid body positions relative to protype at origin
            orientation : quaternions defining rotation relative to prototype
            protype     : list of points defining the rigid body prototype
            box         : matrix of cell vectors defining the simulation box
            filename    : name of file to write xyz
    '''     

    Nbodies = len(position)           # Number of rigid bodies
    Nbeads  = Nbodies*len(prototype)  # Number of points within each rigid body

    # Open output file and write header
    outfile = open(filename,'w')   
    outfile.write(str(Nbeads)+'\n')
    
    # HOOMD stores the simulation cell vectors as the transpose of the cell vectors
    box = np.transpose(box)
    
    cellstring = [str(box[0][0]) , str(box[0][1]), str(box[0][2]),
                  str(box[1][0]) , str(box[1][1]), str(box[1][2]),
                  str(box[2][0]) , str(box[2][1]), str(box[2][2])]
    
    outfile.write(' '.join(cellstring)+'\n')

    # Calculate individual bead positions and write to file
    for idim, refpos in enumerate(position):
        for ibead , _ in enumerate(prototype):

            quat = np.array(orientation[idim])
            vect = np.array(prototype[ibead])

            beadpos =  np.array(refpos) + quat_conjugate_q_with_v(quat, vect)

            outfile.write('C ' + ' '.join([f"{x:.8f}" for x in beadpos]) + '\n')

    outfile.close()

# def write_psf(position, prototype):
#     ''' Writes a psf file for linear chain molecules.
#         Filename is chain.psf.
#     '''

#     Nchains = len(position)
#     Nbeads  = len(prototype)

#     dcd.write_psf(Nchains, Nbeads)


# def write_dcd_header(position, prototype):
#     ''' Writes a dcd file hheadefor linear chain molecules 
#         Filename is chain.dcd
#     '''

#     Nchains = len(position)
#     Nbeads  = len(prototype)

#     dcd.write_dcd_header(Nchains, Nbeads)

#     print("Written DCD header for : ",Nchains," chains with ",Nbeads, " beads per chain")

# def write_dcd_snapshot(position, orientation, prototype, box):
#     ''' Appends a shapshot to chain.dcd, constructed from the position, 
#         orienation and rigid body prototype supplied.  

#         paramters:
#             positions   : rigid body positions relative to protype at origin
#             orientation : quaternions defining rotation relative to prototype
#             protype     : list of points defining the rigid body prototype
#             box         : matrix of cell vectors defining the simulation box
#     '''     

#     Nbodies = len(position)                     # Number of rigid bodies
#     Nbeads  = len(prototype)                    # Number of points within each rigid body

#     # Allocate coordinate array for all beads
#     rbodies = np.empty([Nbodies, Nbeads, 3], dtype=np.float64)

#     # HOOMD stores the simulation cell vectors as the transpose of the cell vectors
#     hmatrix = np.transpose(box)

#     # Calculate individual bead positions and add to array
#     for idim, refpos in enumerate(position):
#         for ibead , _ in enumerate(prototype):

#             quat = np.array(orientation[idim])
#             vect = np.array(prototype[ibead])

#             beadpos = np.array(refpos) + quat_conjugate_q_with_v(quat, vect)
#             rbodies[idim, ibead,0] = beadpos[0]
#             rbodies[idim, ibead,1] = beadpos[1]
#             rbodies[idim, ibead,2] = beadpos[2]


#     # Invoke dcd_tools
#     dcd.write_dcd_snapshot(rbodies,hmatrix)

def minimum_image(r1, r2, box):
    ''' For the supercell vectors defined in box, use minimum image convention
        to find the shortest vector between the two position vectors 
        r1 and r2. 
        
        parameters:
            r1          : first position vector
            r2          : second position vector
            box         : matrix of cell vectors defining the simulation box        

    '''

    hmatrix = np.transpose(box)

    # Calculate inverse matrix of cell vectors
    h_inverse = np.linalg.inv(hmatrix)

    dr = r2 - r1

    sr = np.matmul(h_inverse, dr)
    sr = [ x - np.floor(x+0.5) for x in sr ]

    return np.matmul(hmatrix, sr)
    
    
def check_sphere_overlaps(position, orientation, prototype, box):
    ''' For a configuration consisting entirely of rigid bodies constructed as a 
        union of hard sphere, sanity check that there are no overlaps between
        pairs of spheres on different bodies.

        parameters:
            positions   : rigid body positions relative to protype at origin
            orientation : quaternions defining rotation relative to prototype
            protype     : list of points defining the rigid body prototype
            box         : matrix of cell vectors defining the simulation box

    '''     

    Nbodies = len(position)            # Number of rigid bodies
    Nbeads  = len(prototype.positions) # Number of points within each rigid body

    # Allocate coordinate array for all beads
    rbodies = np.empty([Nbodies, Nbeads, 3], dtype=np.float64)

    # HOOMD stores the simulation cell vectors as the transpose of the cell vectors
    hmatrix = np.transpose(box)

    overlaps = []
    
    # Brute force O(N^2) check for overlaps
    for idim, ipos in enumerate(position):
        for ibead, _ in enumerate(prototype.positions):

            quat = np.array(orientation[idim])
            vect = np.array(prototype.positions[ibead])

            ibeadpos = np.array(ipos) + quat_conjugate_q_with_v(quat, vect)

            for jdim in range(idim+1, len(position)):

                jpos = position[jdim]
                
                for jbead, _ in enumerate(prototype.positions):

                    quad = np.array(orientation[jdim])
                    vect = np.array(prototype.positions[jbead])

                    jbeadpos = np.array(jpos) + quat_conjugate_q_with_v(quad,vect)

                    minvec  = minimum_image(ibeadpos, jbeadpos, box)
                    
                    mindist = np.sqrt(minvec.dot(minvec))

                    if mindist < prototype.sigma:

                        print("Overlap detected between beads:")
                        print(f"  Dimer 1: Body {idim}, Bead {ibead}, Position {ibeadpos}")
                        print(f"  Dimer 2: Body {jdim}, Bead {jbead}, Position {jbeadpos}")
                        print(f"  Minimum Distance: {mindist}")

                        overlaps.append( ( (idim,ibead), (jdim, jbead), mindist ) )
                

    return overlaps

                        
