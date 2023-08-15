import pyrosetta
import pyrosetta.toolbox.numpy_utils as np_utils
import numpy as np
import os
import sys
import copy
import argparse

_DEBUG = False


#====================== pose transformation functions ===========================>

def numpy_to_rosetta(np_arr):
    '''Pack values in a numpy data structure into the analogous Rosetta
    data structure.

    Args:
        np_arry (np.mat or np.array): One- (1 x 3) or two-dimensional (3 x 3)
            numpy matrix

    Returns:
        The values in the appropriate Rosetta container
            (numeric.xyzVector_double_t or numeric.xyzMatrix_double_t)

    '''
    # start off assuming a 1D array
    dim = 1

    # ensure that we are in 3D space
    if np.shape(np_arr)[1] != 3:
        dim = -999
    elif np.shape(np_arr)[0] == np.shape(np_arr)[1]:
        dim = 2
        
    if dim == 1:
        # handle the 1D case
        ros_container = pyrosetta.rosetta.numeric.xyzVector_double_t(0.)
        ros_container.x = np_arr[0, 0]
        ros_container.y = np_arr[0, 1]
        ros_container.z = np_arr[0, 2]

        return ros_container

    elif dim == 2:
        # handle the 2D case
        ros_container = pyrosetta.rosetta.numeric.xyzMatrix_double_t(0.)
        ros_container.xx = np_arr[0, 0]
        ros_container.xy = np_arr[0, 1]
        ros_container.xz = np_arr[0, 2]

        ros_container.yx = np_arr[1, 0]
        ros_container.yy = np_arr[1, 1]
        ros_container.yz = np_arr[1, 2]

        ros_container.zx = np_arr[2, 0]
        ros_container.zy = np_arr[2, 1]
        ros_container.zz = np_arr[2, 2]

        return ros_container

    # get out of here!
    raise ValueError('Packing {}-dimensional numpy arrays '.format(dim) +
                     'into Rosetta containers is not currently supported')

def rotate_pose(p, R):
    '''Apply a rotation matrix to all of the coordinates in a Pose.

    Args:
        p (Pose): The Pose instance to manipulate
        R (np.mat): A rotation matrix to apply to the Pose coordinates

    Returns:
        None. The input Pose is manipulated.

    '''
    # t must be an xyzMatrix_Real
    for i in range(1, p.size() + 1):
        for j in range(1, p.residue(i).natoms() + 1):
            v = p.residue(i).atom(j).xyz()

            x = R.xx * v.x + R.xy * v.y + R.xz * v.z
            y = R.yx * v.x + R.yy * v.y + R.yz * v.z
            z = R.zx * v.x + R.zy * v.y + R.zz * v.z

            p.residue(i).atom(j).xyz(pyrosetta.rosetta.numeric.xyzVector_double_t(x, y, z))

def translate_pose(p, t):
    '''Apply a translation to all of the coordinates in a Pose.

    Args:
        p (Pose): The Pose instance to manipulate
        t (np.array): A vector to add to the Pose coordinates

    Returns:
        None. The input Pose is manipulated.

    '''
    # t must be an xyzVector_double_t
    for i in range(1, p.size() + 1):
        for j in range(1, p.residue(i).natoms() + 1):
            p.residue(i).atom(j).xyz(p.residue(i).atom(j).xyz() + t)



def rotation_matrix_from_vectors(vec1, vec2):
    '''
    from: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
     Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    '''
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_matrix_from_axis_vec_and_theta(axis_vec, theta, radian=False):
    '''
        compute and set R from rotation axis: axis_vec and rotation angle: theta
        https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    '''
    axis_vec = np.array(axis_vec)
    assert(axis_vec.shape==(3,)) 
    if not radian:
        theta = np.radians(theta)
    ux, uy, uz = axis_vec/ np.linalg.norm(axis_vec)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)        
    rot_mtx_np = np.zeros(9).reshape(3,3)

    rot_mtx_np[0] = cos_theta+(ux**2)*(1-cos_theta), \
                    ux*uy*(1-cos_theta)-uz*sin_theta, \
                    ux*uz*(1-cos_theta)+uy*sin_theta

    rot_mtx_np[1] = uy*ux*(1-cos_theta)+uz*sin_theta, \
                    cos_theta+(uy**2)*(1-cos_theta), \
                    uy*uz*(1-cos_theta)-ux*sin_theta

    rot_mtx_np[2] = uz*ux*(1-cos_theta)-uy*sin_theta, \
                    uz*uy*(1-cos_theta)+ux*sin_theta, \
                    cos_theta+(uz**2)*(1-cos_theta)
    return rot_mtx_np


def compute_rigid_3D_transform(A, B):
    '''
         https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
         
         Input: A, B: expect 3xN matrix of points
         Returns R,t for the transform of A->B
         R = 3x3 rotation matrix
         t = 3x1 column vector
    '''
    A, B = np.mat(A), np.mat(B)
    assert(A.shape==B.shape)
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))
    [num_rows, num_cols] = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))
    H = Am * np.transpose(Bm)
    # sanity check
    #if np.linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    t = -R*centroid_A + centroid_B
    return np.array(R), np.array(t) # conver np.mat to np.array


def compute_rigid_3D_transform_for_poses(mobile_pose, mobile_reslist, 
                                         target_pose, target_reslist,
                                         atomtype=['CA']):
    mobile_coords = np.zeros(len(atomtype)*len(mobile_reslist)*3).reshape(len(atomtype)*len(mobile_reslist), 3)
    for i, resid in enumerate(mobile_reslist):
        for j, atm in enumerate(atomtype):
            mobile_coords[i*len(atomtype)+j] = mobile_pose.residue(resid).xyz(atm) 
    
    target_coords = np.zeros(len(atomtype)*len(target_reslist)*3).reshape(len(atomtype)*len(target_reslist), 3)
    for i, resid in enumerate(target_reslist):
        for j, atm in enumerate(atomtype):
            target_coords[i*len(atomtype)+j] = target_pose.residue(resid).xyz(atm)     
    
    return compute_rigid_3D_transform(mobile_coords.T, target_coords.T)




class rigid_3D_transform:
    '''
        for storing rotation matrix/axis_angle and translation vector
        
        two ways of rotation:
            1. rotate by rotation matrix
            2. rotate by newly defined origin, axis vector and rotation angle
    '''
    
    def __init__(self, rotate_orig_=np.zeros(3), axis_vec_=None, 
                 theta_=None, R_=None, t_=None, radian=True):
        '''
            axis_vec:     (ndarray) of rotation axis vector
            theta:        rotation degree (always stored as in radian)
            R:            (ndarray) rotation matrix
            t:            (ndarray) translation vector
            radian:       rotation angle unit in radian (false if in degreee)
            
            axis_vec and theta overwritten if R specified
        '''
        assert(rotate_orig_.shape==(3,))
        self.rotate_orig = rotate_orig_
        
        if axis_vec_ is not None:
            assert(axis_vec_.shape==(3,))
        self.axis_vec = axis_vec_
        
        if not radian and theta_ is not None:
            self.theta = np.radians(theta_)
        else:
            self.theta = theta_

        if R_ is not None:
            assert(R_.shape==(3,3))
            if self.axis_vec is not None or self.theta is not None:
                print('Warning: overwriting axis_vec and theta, because R is already specified!')
                self.axis_vec, self.theta = None, None
        self.R = R_
        
        self.t = t_

    
    def set_axis_vec_and_theta(self, axis_vec_, theta_, radian=True):
        if axis_vec_ is not None:
            assert(axis_vec_.shape==(3,))
        self.axis_vec = axis_vec_
        if not radian and theta_ is not None:
            self.theta = np.radians(theta_)
        else:
            self.theta = theta_        
    
    
    def update_R_by_axis_vec_and_theta(self):
        '''
            compute and set R from self.axis_vec and self.theta
            https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        '''
        assert(self.axis_vec is not None and self.axis_vec.shape==(3,)) 
        assert(self.theta is not None)
        ux, uy, uz = self.axis_vec/ np.linalg.norm(self.axis_vec)
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)        
        rot_mtx_np = np.zeros(9).reshape(3,3)

        rot_mtx_np[0] = cos_theta+(ux**2)*(1-cos_theta), \
                        ux*uy*(1-cos_theta)-uz*sin_theta, \
                        ux*uz*(1-cos_theta)+uy*sin_theta

        rot_mtx_np[1] = uy*ux*(1-cos_theta)+uz*sin_theta, \
                        cos_theta+(uy**2)*(1-cos_theta), \
                        uy*uz*(1-cos_theta)-ux*sin_theta

        rot_mtx_np[2] = uz*ux*(1-cos_theta)-uy*sin_theta, \
                        uz*uy*(1-cos_theta)+ux*sin_theta, \
                        cos_theta+(uz**2)*(1-cos_theta)
        self.R = rot_mtx_np

    
    def update_axis_vec_and_theta_by_R(self, epsilon=0.01, epsilon2=0.1, radian=True):
        '''
            compute and set self.axis_vec and self.theta by self.R
            http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
            CAUTION: singularity when angle=0 or 180

            epsilon: margin to allow for rounding errors
            epsilon2: margin to distinguish between 0 and 180 degrees singularities
        '''
        m = self.R
        assert(m.shape==(3,3))
        # check and deal with singularity
        if np.abs(m[0][1]-m[1][0]) < epsilon and np.abs(m[0][2]-m[2][0]) < epsilon \
            and np.abs(m[1][2]-m[2][1]) < epsilon:
            print('Warning: singularity found converting rotation matrix to axis and angle')
            # case of angle=0
            if np.abs(m[0][0]+m[1][1]+m[2][2]-3) < epsilon2:
                # this singularity is identity matrix so angle = 0, arbitrary axis (z axis)
                self.axis_vec = np.array([0,0,1])
                self.theta = 0 
                return None
            # case of angle=180
            xx = (m[0][0]+1)/2
            yy = (m[1][1]+1)/2
            zz = (m[2][2]+1)/2
            xy = (m[0][1]+m[1][0])/4
            xz = (m[0][2]+m[2][0])/4
            yz = (m[1][2]+m[2][1])/4
            r2_2 = np.sqrt(2)/2
            if xx > yy and xx > zz: 
                x, y, z = 0, r2_2, r2_2 if xx < epsilon else np.sqrt(xx), xy/x, xz/x
            elif yy > zz: 
                x, y, z = r2_2, 0, r2_2 if yy < epsilon else xy/y, np.sqrt(yy), yz/y
            else:
                x, y, z = r2_2, r2_2, 0 if zz < epsilon else xz/z, yz/z, np.sqrt(zz)
            theta_ = np.pi if radian else 180
            self.axis_vec = np.array([x,y,z])
            self.theta = theta_
            return None
        theta_ = np.arccos( (m[0][0]+m[1][1]+m[2][2]-1)/2 )
        theta_ = theta_ if radian else np.degrees(theta_)
        s = np.sqrt((m[2][1] - m[1][2])**2+(m[0][2] - m[2][0])**2+(m[1][0] - m[0][1])**2)
        s = s if s > 0.0001 else 1 # prevent division by 0
        x = (m[2][1] - m[1][2])/s
        y = (m[0][2] - m[2][0])/s
        z = (m[1][0] - m[0][1])/s
        self.axis_vec = np.array([x,y,z])
        self.theta = theta_


    
    
class rigid_3D_transform_pose(rigid_3D_transform):
    '''
        transform operations for rosetta pose
    '''
    def __init__(self, pose_=None,
                 rotate_orig_=np.zeros(3), axis_vec_=None, 
                 theta_=None, R_=None, t_=None, radian=True):
        super().__init__(rotate_orig_=rotate_orig_, axis_vec_=axis_vec_, 
                         theta_=theta_, R_=R_, t_=t_, radian=radian)
        self.pose = pose_    

    def rotate(self, rotate_by_R=True):
        assert(self.pose is not None)
        if rotate_by_R:
            if self.R is not None:
                rotate_pose(self.pose, numpy_to_rosetta(self.R))
            else:
                print('Error: trying to rotate by R without setting R!')
                raise ValueError
        else:
            # rotate by rotate_orig, axis_vec and theta
            #   1. align rotate_orig to origin (0,0,0)
            #   2. rotate by R after updating with axis_vec and theta
            #   3. move rotate_orig back to initial position
            assert(self.axis_vec is not None and self.theta is not None)
            self.update_R_by_axis_vec_and_theta()
            translate_pose(self.pose, pyrosetta.rosetta.numeric.xyzVector_double_t(*(-1*self.rotate_orig)))
            rotate_pose(self.pose, numpy_to_rosetta(self.R))
            translate_pose(self.pose, pyrosetta.rosetta.numeric.xyzVector_double_t(*self.rotate_orig))
        
    def translate(self):
        assert(self.pose is not None)
        assert(self.t is not None)
        translate_pose(self.pose, pyrosetta.rosetta.numeric.xyzVector_double_t(*self.t))

    
    
    
class ParamHelix(rigid_3D_transform_pose):
    
    
    def __init__(self, helix_len_=20, dummy_input_=None, 
                 crick_param_file_=None, residue_name3_='ALA',
                 pose_=None,
                 rotate_orig_=np.zeros(3), axis_vec_=None, 
                 theta_=None, R_=None, t_=None, radian=True):
        super().__init__(pose_=pose_,
                         rotate_orig_=rotate_orig_, axis_vec_=axis_vec_, 
                         theta_=theta_, R_=R_, t_=t_, radian=radian)
        self.helix_len = helix_len_
        self.dummy_input = dummy_input_
        self.crick_param_file = crick_param_file_
        self.residue_name3 = residue_name3_
    
    
    def generate_pose(self):
        if self.dummy_input is not None:
            self.pose = pyrosetta.pose_from_file(self.dummy_input)
        else:
            self.pose = pyrosetta.pose_from_sequence('A')
        mbh = pyrosetta.rosetta.protocols.helical_bundle.MakeBundleHelix()
        resname_vec = pyrosetta.rosetta.utility.vector1_std_string()
        resname_vec.append(self.residue_name3)
        mbh.set_residue_name(resname_vec)
        mbh.set_helix_length(self.helix_len)
        if self.crick_param_file is not None:
            mbh.set_minor_helix_params_from_file(self.crick_param_file)
        mbh.apply(self.pose)

        
        

class ParamPose(rigid_3D_transform_pose):

    
    def __init__(self, template_pose_, params_={},
                 pose_=None,
                 rotate_orig_=np.zeros(3), axis_vec_=None, 
                 theta_=None, R_=None, t_=None, radian=True):
        '''
            template_pose (required): template pose for creating new repeat 
            params_: dictionary of parameters, e.g. {'r_z1':0}
        '''
        super().__init__(pose_=pose_,
                         rotate_orig_=rotate_orig_, axis_vec_=axis_vec_, 
                         theta_=theta_, R_=R_, t_=t_, radian=radian)
        self.pose = template_pose_.clone()
        self.params=copy.deepcopy(params_)
    
    
    
# <===================== pose transformation functions ===========================



def get_com(pose, reslist=[], atomtype=['CA']):
    if len(reslist) == 0:
        reslist = list(range(1,pose.size()+1))  
    coords = np.zeros(len(atomtype)*len(reslist)*3).reshape(len(atomtype)*len(reslist), 3)
    for i, resid in enumerate(reslist):
        for j, atm in enumerate(atomtype):
            coords[i*len(atomtype)+j] = pose.residue(resid).xyz(atm)
    return coords.sum(axis=0)/coords.shape[0]


def sample_point_on_sphere_surface(theta_range, phi_range):
    '''
        http://corysimon.github.io/articles/uniformdistn-on-sphere/
        the ranges are 2-element lists: [min, max]
    '''
    this_theta = theta_range[0] + np.random.sample()*(theta_range[1]-theta_range[0])
    this_phi = phi_range[0] + np.random.sample()*(phi_range[1]-phi_range[0])
    x = np.cos(this_theta)*np.sin(this_phi)
    y = np.sin(this_theta)*np.sin(this_phi)
    z = np.cos(this_phi)
    return np.array([x,y,z]), this_theta, this_phi

            
def get_anchor_coordinates_from_pose(p, reslist):
    _bb_atoms = ['N', 'CA', 'C', 'O']
    coords = list()
    for resNo in reslist:
        res = p.residue(resNo)
        # only iterate over relevant atoms
        for i in _bb_atoms:
            coords.append([res.xyz(i).x, res.xyz(i).y, res.xyz(i).z])
    return np.mat(coords)


    
def align_pose_to_anchor_coords(p, target_coord, anchor_coord):
    R, t = np_utils.rigid_transform_3D(anchor_coord, target_coord)
    #np_utils.rotate_pose(p, np_utils.numpy_to_rosetta(R)) # JHL, sth wrong w/ dig's pyrosetta: xx() not callable, but xx directly accessible
    rotate_pose(p, numpy_to_rosetta(R))  # JHL, so I had to rewrite np->rosetta and rotation function to change xx() to xx
    #np_utils.translate_pose(p, np_utils.numpy_to_rosetta(t.T)) # on mac there's no translate_pose
    translate_pose(p, numpy_to_rosetta(t.T)) # so i copied the ones for older rosetta codes
    return p

def get_aligned_repeat_pose(input_pose, target_pose, anchor_resid, target_resid):
    ''' align the anchor_resid residue of a copy of input pose
        to the target_resid residue of the input pose
    '''
    target_coord = get_anchor_coordinates_from_pose(target_pose, [target_resid]) 
    anchor_coord = get_anchor_coordinates_from_pose(input_pose, [anchor_resid])
    new_pose = input_pose.clone()
    new_pose_aligned = align_pose_to_anchor_coords(new_pose, target_coord, anchor_coord)
    return new_pose_aligned
    

    
def poorman_repeat_propagate(pose, repeat_len, num_repeat=4, overhang=1):
    '''
        overhang: index of residue in the 2nd repeat to be used for alignment; should >= 1
    '''

    def _append_residue(pose, source_pose, resid, source_resid):
        if source_pose.chain(source_resid) != source_pose.chain(source_resid-1) \
            or source_pose.residue(source_resid).xyz('CA').distance(source_pose.residue(source_resid-1).xyz('CA')) > 4:
            pose.append_residue_by_jump(source_pose.residue(source_resid), resid-1)
        else:
            pose.append_residue_by_bond(source_pose.residue(source_resid))
        return pose 
    
    assert(pose.size() >= repeat_len+overhang)
    
    # build the 1st repeats
    new_pose = pyrosetta.rosetta.core.pose.Pose()
    new_pose.append_residue_by_jump(pose.residue(1), 1)
    for i in range(2, repeat_len+overhang+1):
        new_pose = _append_residue(new_pose, pose, i, i)
    
    # build the rest repeats
    for rep in range(num_repeat-1):
        current_repeat_pose = get_aligned_repeat_pose(pose, new_pose, overhang, (rep+1)*repeat_len+overhang)
        for i in range(overhang+1,repeat_len+overhang+1):
            # avoid padding overhang into last repeat
            if rep == num_repeat-2 and i > repeat_len:
                break
            else:
                new_pose = _append_residue(new_pose, current_repeat_pose, rep*repeat_len+i, i)
          
    return new_pose

            
def get_angle_from_edge_length(a, b, c):
    '''
    given the length of a triangle, return angle
    a, b are the length of edge that define the angle
    '''
    return np.arccos( (a**2 + b**2 - c**2) / (2*a*b) )





#===================================  symmetric relax/min ================================>>>>>>


# fills in a SymmetryInfo object with the necessary info
def setup_repeat_symminfo(repeatlen, symminfo, nrepeat, base_repeat):
    #print('doing setup_repeat_symminfo')
    base_offset = (base_repeat-1)*repeatlen
    i=1
    while i <= nrepeat: 
        if i == base_repeat: 
            i=i+1
            continue
        offset = (i-1)*repeatlen
        j=1
        while j <= repeatlen:
            symminfo.add_bb_clone(base_offset+j, offset+j )
            symminfo.add_chi_clone( base_offset+j, offset+j )
            j=j+1
        i=i+1


    symminfo.num_virtuals( 1 ) # the one at the end...
    symminfo.set_use_symmetry( True )
    symminfo.set_flat_score_multiply( nrepeat*repeatlen+1, 1 )
    symminfo.torsion_changes_move_other_monomers( True ) # note -- this signals that we are folding between repeats

    ### what is the reason to do this???
    ### If there is a good reason, why not do for repeats after base_repeat???
    ### 
    # repeats prior to base_repeat have score_multiply ==> 0
    """
    i=1
    while i < base_repeat:
        offset = (i-1)*repeatlen
        j=1
        while j <= repeatlen:
            symminfo.set_score_multiply( offset+j, 0 )
            j=j+1
        i=i+1
    """
    symminfo.update_score_multiply_factor()
    #print('finished setup_repeat_symminfo')


# sets up a repeat pose, starting from a non-symmetric pdb with nres=repeatlen*nrepeat
def setup_repeat_pose(pose, numb_repeats_=4, base_repeat=2):
    #print('doing setup_repeat_pose')
    if pyrosetta.rosetta.core.pose.symmetry.is_symmetric(pose):
        return False # not to begin with...
    repeatlen = int(pose.size()/numb_repeats_)
    nrepeat = numb_repeats_
    
    if not nrepeat * repeatlen == pose.size():
        return False

    if not base_repeat > 1:
        return False

    nres_protein = nrepeat * repeatlen
    pyrosetta.rosetta.core.pose.remove_upper_terminus_type_from_pose_residue( pose, pose.size() )
    vrtrsd = pyrosetta.rosetta.core.conformation.Residue(pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue
           (pyrosetta.rosetta.core.pose.get_restype_for_pose(pose, "VRTBB" )))
    pose.append_residue_by_bond( vrtrsd, True ) # since is polymer...
    #pose.append_residue_by_jump( vrtrsd, True )
    pose.conformation().insert_chain_ending( nres_protein )
    f = pyrosetta.rosetta.core.kinematics.FoldTree( pose.size() )
    f.reorder( pose.size() )
    pose.fold_tree( f )
    symminfo = pyrosetta.rosetta.core.conformation.symmetry.SymmetryInfo()
    setup_repeat_symminfo( repeatlen, symminfo, nrepeat, base_repeat )

    # now make symmetric
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_pose(pose, symminfo)
    
    if not pyrosetta.rosetta.core.pose.symmetry.is_symmetric(pose):
        return False

    base_offset = (base_repeat-1)*repeatlen
    """
    for i in range(0,nrepeat):
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            pose.set_phi  ( j+i*repeatlen, pose.phi  (pos) )
            pose.set_psi  ( j+i*repeatlen, pose.psi  (pos) )
            pose.set_omega( j+i*repeatlen, pose.omega(pos) )
            j=j+1
       
    for i in range(0,nrepeat): 
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            oldrsd = pyrosetta.rosetta.core.conformation.Residue( pose.residue(pos).clone() )
            pose.replace_residue( j+i*repeatlen, oldrsd, False )
            j=j+1
    """
    #print('finished setup_repeat_pose')


def setup_movemap(pose, bblist=[], chilist=[]):
    #print('doing setup_movemap')
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    if len(bblist) == 0 and len(chilist) == 0:
        mm.set_chi( True )
        mm.set_bb( True )        
    else:
        mm.set_chi( False )
        mm.set_bb( False )
        for resid in range(1, pose.size()+1):
            if resid in bblist:
                mm.set_bb( resid, True )
                mm.set_chi( resid, True )
            elif resid in chilist:
                mm.set_chi( resid, True )
    mm.set_jump( True )
    mm.set_bb ( pose.size(), False ) # # for the virtual residue?
    mm.set_chi( pose.size(), False ) # for the virtual residue?
    #print('finished setup_movemap')
    return mm

def seal_jumps(pose):   
    #print('doing seal_jumps')
    i=1
    while i <= pose.size():
        if pose.residue_type(i).name() == "VRTBB":
            pose.conformation().delete_residue_slow(i)
        i=i+1
    ii=1
    while ii <= pose.size()-1:
        if ( pose.residue( ii ).has_variant_type( pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER ) 
            and pose.residue( ii+1 ).has_variant_type( pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER ) ):
            pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( pose, pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER, ii )
            pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( pose, pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER, ii+1 )
        ii=ii+1
    ft = pose.fold_tree()
    ft.clear()
    ft.add_edge(1,pose.size(),-1)
    pose.fold_tree(ft)
    #print('finished seal_jumps')


def relax_pose(pose, cartesian_=False, bblist=[], chilist=[], rmsd_check_resids=[]):
    #print('doing relax_pose')
    relax_iterations_ = 1
    pdb_pose = pose.clone()
    s = pyrosetta.get_score_function()
    sf = pyrosetta.rosetta.core.scoring.symmetry.symmetrize_scorefunction(s)        
    
    sf.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 0.1)
    fastrelax = pyrosetta.rosetta.protocols.relax.FastRelax( sf , relax_iterations_ )
    #fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.ramp_down_constraints(False)
    fastrelax.min_type('lbfgs_armijo_nonmonotone')
    if cartesian_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0) # has to be zero
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_angle, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_length, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_torsion, 0.5) # what are good values
        fastrelax.cartesian(True)
        fastrelax.minimize_bond_angles(True)
        fastrelax.minimize_bond_lengths(True)
    movemap = setup_movemap(pose, bblist, chilist)
    
   
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_movemap(pose,movemap)
    #movemap.clear() #for testing purposes only, prevents relax from doing anything
    fastrelax.set_movemap( movemap )
    pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    fastrelax.apply( pose )
    #rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(pose, pdb_pose)
    if len(rmsd_check_resids) > 0:
        rmsd = rmsd_by_ndxs_atoms(pose, rmsd_check_resids[0], rmsd_check_resids[-1], pdb_pose, rmsd_check_resids[0], rmsd_check_resids[-1])
        #print("rmsd change from relax: ",rmsd)
    #print('finished relax_pose')


def minimize_pose(pose, cartesian_=False, coordcst_=False):
    #print('doing minimize_pose')
    movemap = setup_movemap(pose)
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_movemap(pose,movemap)
    s = pyrosetta.get_score_function()
    sf = pyrosetta.rosetta.core.scoring.symmetry.symmetrize_scorefunction(s)    
    if coordcst_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)
        pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    
    use_nblist = True
    deriv_check = True
    deriv_check_verbose = False
    #min_mover = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymMinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.1,use_nblist,
    #                                                                      deriv_check,deriv_check_verbose)
    min_mover = pyrosetta.rosetta.protocols.simple_moves.symmetry.SymMinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.1,use_nblist,
                                                                          deriv_check,deriv_check_verbose)
    min_mover.max_iter(1)
    #min_mover.min_type('lbfgs_armijo_nonmonotone')
      
    if cartesian_:
        min_mover.cartesian(True)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0)

    min_mover.apply( pose )
    #print('finished minimize_pose')

def RepeatProteinRelax_apply(pose, modify_symmetry_and_exit_=False, remove_symm_=False, minimize_=False, cartesian_=False, bblist=[], chilist=[], rmsd_check_resids=[], num_repeats=4, base_repeat=2):
    #print('doing RepeatProteinRelax_apply')
    if modify_symmetry_and_exit_ and remove_symm_:
        pyrosetta.rosetta.core.pose.symmetry.make_asymmetric_pose(pose)
        seal_jumps(pose)
        return True
    setup_repeat_pose(pose, numb_repeats_=num_repeats, base_repeat=base_repeat)
    setup_movemap(pose, bblist, chilist);
    if modify_symmetry_and_exit_ and not remove_symm_:
        return True
    if minimize_:
        minimize_pose(pose, cartesian_)
    else:
        relax_pose(pose, cartesian_, bblist, chilist, rmsd_check_resids)
    pyrosetta.rosetta.core.pose.symmetry.make_asymmetric_pose(pose)
    seal_jumps(pose)

#<<<<<<=============================  symmetric relax/min ======================================




def _config_my_task_factory(added_residues, allowed_aa='ACDEFGHIKLMNPQRSTVWY', include_neighbor_repack=True, neighbor_dist=6.0):

    number_of_residues = len(added_residues)

    loop_residues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    for res_num in added_residues:
        loop_residues.append_index(res_num)

    # only use a subset of (canonical) AAs when designing loop residues
    restrict_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    #restrict_res.aas_to_keep('AVLIFMSTNQYW')
    #restrict_res.aas_to_keep('AP') # bb check
    #restrict_res.aas_to_keep('RKDEQNHSTYW') # polar residues
    restrict_res.aas_to_keep(allowed_aa) 
    allow_design_on_loop_res = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        restrict_res, loop_residues)

    nbrhood_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
        loop_residues, neighbor_dist, False)
    nbrhood_sel_inclusive = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
        loop_residues, neighbor_dist, True)


    # disable design for all residues outside the loop
    restrict_nbrs_to_repack = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), nbrhood_sel, False)  # not flip the selection

    # don't pack residues more than 10. A from the loop
    only_repack_neighborhood = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), nbrhood_sel_inclusive, True)

    # add task ops to a TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(allow_design_on_loop_res)
    tf.push_back(only_repack_neighborhood)
    tf.push_back(restrict_nbrs_to_repack)
    return tf


def _config_my_task_factory_layer(core_selector, boundary_selector, surface_selector, helix_ncap_selector=None, 
            core_aa='ACDEFGHIKLMNPQRSTVWY', boundary_aa='ACDEFGHIKLMNPQRSTVWY', surface_aa='ACDEFGHIKLMNPQRSTVWY', helix_ncap_aa='DNST', include_neighbor_repack=True, neighbor_dist=6.0):
    
    core_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    core_res.aas_to_keep(core_aa) 
    core_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(core_res, core_selector)

    boundary_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    boundary_res.aas_to_keep(boundary_aa) 
    boundary_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(boundary_res, boundary_selector)

    surface_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    surface_res.aas_to_keep(surface_aa) 
    surface_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(surface_res, surface_selector)

    core_boundary_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_selector, boundary_selector)
    core_boundary_surface_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_boundary_selector, surface_selector)

    helix_ncap_res_subset = None
    if helix_ncap_selector != None:
        helix_ncap_res = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
        helix_ncap_res.aas_to_keep(helix_ncap_aa) 
        helix_ncap_res_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(helix_ncap_res, helix_ncap_selector)        
        core_boundary_surface_helixcap_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_boundary_surface_selector, helix_ncap_selector)
        neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_helixcap_selector, neighbor_dist, False) # not including selector residues
        design_n_neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_helixcap_selector, neighbor_dist, True) # including selector residues
    else:
        neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_selector, neighbor_dist, False) # not including selector residues
        design_n_neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(core_boundary_surface_selector, neighbor_dist, True) # including selector residues

    repack_nbr_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), neighbor_selector)    

    freeze_rest_subset = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), design_n_neighbor_selector, True) # flip the selection

    # add task ops to a TaskFactory
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(core_res_subset)
    tf.push_back(boundary_res_subset)
    tf.push_back(surface_res_subset)
    if helix_ncap_res_subset != None:
        tf.push_back(helix_ncap_res_subset)        
    tf.push_back(repack_nbr_subset)
    tf.push_back(freeze_rest_subset)
    return tf    



def _config_my_move_map(p, input_obj, include_neighbor=True, neighbor_dist=6.0):
    mm = pyrosetta.MoveMap()
    if isinstance(input_obj, list):
        ind_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector('{}'.format(','.join([str(x) for x in input_obj])))
    else:
        # input is already a selector
        ind_selector = input_obj
    ind = ind_selector.apply(p)
    mm.set_bb(ind)
    if include_neighbor:
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(ind_selector, neighbor_dist)
        nbr = nbr_selector.apply(p)
        #print(nbr)
        mm.set_chi(nbr)
    else:
        mm.set_chi(ind)

    return mm


def fast_design_layer_relaxscript(p, sf, core_selector, boundary_selector, surface_selector, helix_ncap_selector=None,
                                    core_aa='ACDEFGHIKLMNPQRSTVWY', boundary_aa='ACDEFGHIKLMNPQRSTVWY', surface_aa='ACDEFGHIKLMNPQRSTVWY', helix_ncap_aa='DNST', rmsd_check_resids=[], include_neighbor=True, neighbor_dist=6.0,
                                    relaxscript='/home/jianghl/software/Rosetta/main/database/sampling/relax_scripts/MonomerDesign2019.txt'):

    pose = p.clone()
    fd = pyrosetta.rosetta.protocols.denovo_design.movers.FastDesign()
    fd.set_task_factory(_config_my_task_factory_layer(core_selector, boundary_selector, surface_selector, helix_ncap_selector, core_aa, boundary_aa, surface_aa, helix_ncap_aa, include_neighbor_repack=include_neighbor, neighbor_dist=neighbor_dist))
    fd.set_scorefxn(sf)
    core_boundary_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_selector, boundary_selector)
    core_boundary_surface_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(core_boundary_selector, surface_selector)
    fd.set_movemap(_config_my_move_map(p, core_boundary_surface_selector, include_neighbor=include_neighbor, neighbor_dist=neighbor_dist))


    relaxscript = pyrosetta.rosetta.std.vector_std_string()
    relaxscript.append('repeat {}'.format(pyrosetta.rosetta.basic.options.get_integer_option('relax:default_repeats')))
    fin_rs = open(relaxscript,'r')
    for line in fin_rs.readlines()[1:]:
        relaxscript.append(line.strip())
    #print(relaxscript)
    fd.set_script_from_lines(relaxscript)

    fd.apply(p)
    if len(rmsd_check_resids) > 0:
        rmsd = rmsd_by_ndxs_atoms(p, rmsd_check_resids[0], rmsd_check_resids[-1], pose, rmsd_check_resids[0], rmsd_check_resids[-1])
        #print("rmsd change from design: ",rmsd)
    return None


def pack(pose, sf, pack_reslist, allowed_aa='ACDEFGHIKLMNPQRSTVWY'):
    pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sf)
    pack_rotamers.task_factory(_config_my_task_factory(pack_reslist, allowed_aa=allowed_aa))
    pack_rotamers.apply(pose)
    return None

def sym_pack(pose, sf, pack_reslist, allowed_aa='ACDEFGHIKLMNPQRSTVWY'):
    pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymPackRotamersMover(sf)
    pack_rotamers.task_factory(_config_my_task_factory(pack_reslist, allowed_aa=allowed_aa))
    pack_rotamers.apply(pose)
    return None

def sym_min(pose, sf, cartesian_=False, coordcst_=False):
    #print('doing minimize_pose')
    movemap = setup_movemap(pose)
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_movemap(pose,movemap)
    if coordcst_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)
        pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    
    use_nblist = True
    deriv_check = True
    deriv_check_verbose = False
    #min_mover = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymMinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.1,use_nblist,
    #                                                                      deriv_check,deriv_check_verbose)
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.symmetry.SymMinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.1,use_nblist,
                                                                          deriv_check,deriv_check_verbose)
    min_mover.max_iter(1)
    #min_mover.min_type('lbfgs_armijo_nonmonotone')
      
    if cartesian_:
        min_mover.cartesian(True)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0)

    min_mover.apply( pose )
    #print('finished minimize_pose')


#### add helical capping calculate helix-Zaxis vectors


def align_cap_pose_to_anchor_coords(cap, coords, ncap=True):
    # ncap: True (ncap), False (ccap)
    if ncap:
        moveable_coords = get_anchor_coordinates_from_pose(cap, [cap.size()-1])
    else:
        moveable_coords = get_anchor_coordinates_from_pose(cap, [2])
    R, t = np_utils.rigid_transform_3D(moveable_coords, coords)
    #np_utils.rotate_pose(p, np_utils.numpy_to_rosetta(R)) # JHL, sth wrong w/ dig's pyrosetta: xx() not callable, but xx directly accessible
    rotate_pose(cap, numpy_to_rosetta(R))  # JHL, so I had to rewrite np->rosetta and rotation function to change xx() to xx
    #np_utils.translate_pose(cap, np_utils.numpy_to_rosetta(t.T))
    translate_pose(cap, numpy_to_rosetta(t.T))
    return cap

def generate_cap(phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=True, aacomp_cap_pro_file="-1"):
    cap_seq = aa*(len(phipsi)+2)
    cap = pyrosetta.pose_from_sequence(cap_seq) # two more residues serve as dummy termini for phipsi definition
    for i in range(1,len(phipsi)+1):
        cap.set_phi(i+1,phipsi[i-1][0])
        cap.set_psi(i+1,phipsi[i-1][1])
        # omega is 180 by default

    #
    # TODO
    # get the binselector to work!!
    #
    if insert_GLy: # put pro and gly in if its their bin
        pack_reslist = []
        for i in range(1,len(phipsi)+1):
            # gly
            if phipsi[i-1][0] > 0: # CAUTION!! hard coded way of identify gly bin 
                mt = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(i+1, 'GLY')
                mt.apply(cap)
            else:
                pack_reslist.append(i+1)
    else:
        pack_reslist = [x+1 for x in range(1, len(phipsi)+1)]

    if insert_Pro:        

        if aacomp_cap_pro_file != "-1":
            p_bin = pyrosetta.rosetta.core.select.residue_selector.BinSelector()
            p_bin.set_bin_params_file_name('PRO_DPRO')
            p_bin.set_bin_name('LPRO')
            p_bin.initialize_and_check()
            aacomp_pro = pyrosetta.rosetta.protocols.aa_composition.AddCompositionConstraintMover()
            aacomp_pro.create_constraint_from_file(aacomp_cap_pro_file)
            aacomp_pro.add_residue_selector(p_bin)
            aacomp_pro.apply(cap) 

        # pro (need the bin selector here)
        #sf.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.2)

        ## old packer
        #pack_rotamers = pyrosetta.rosetta.protocols.simple_moves.PackRotamersMover(sf_cap)
        #pack_rotamers.task_factory(_config_my_task_factory(pack_reslist, allowed_aa='AP'))

        pack_rotamers = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sf_cap)
        pack_rotamers.task_factory(_config_my_task_factory(pack_reslist, allowed_aa='AP'))
        pack_rotamers.apply(cap)

    return cap

def prepend_n_cap(p, cap):
    # prepend ncap to the beginning of the pose p
    p_anchor_coord = get_anchor_coordinates_from_pose(p, [1])
    align_cap_pose_to_anchor_coords(cap, p_anchor_coord, ncap=True)
    new_p = pyrosetta.rosetta.core.pose.Pose()
    new_p.append_residue_by_jump(cap.residue(2), 1)
    for i in range(3,cap.size()):
        new_p.append_residue_by_bond(cap.residue(i))
    for i in range(2, p.size()+1): # skipping the overlapping resiude on the scaffold
        if 'Nterm' in p.residue(i).name():
            new_p.append_residue_by_jump(p.residue(i),new_p.size())
        else:
            new_p.append_residue_by_bond(p.residue(i))    
    return new_p

def insert_n_cap(p, anchor_pos, cap):
    # inserting ncap to the tail side of the gap in pdb (closer to pdb's cterm)
    # assuming ncap is inserted before ccap!
    p_anchor_coord = get_anchor_coordinates_from_pose(p, [anchor_pos+1])
    align_cap_pose_to_anchor_coords(cap, p_anchor_coord, ncap=True)
    new_p = pyrosetta.rosetta.core.pose.Pose()
    new_p.append_residue_by_jump(p.residue(1), 1)
    for i in range(2,anchor_pos+1):
        new_p.append_residue_by_bond(p.residue(i))
    new_p.append_residue_by_jump(cap.residue(2), anchor_pos)
    for i in range(3,cap.size()):
        new_p.append_residue_by_bond(cap.residue(i))
    for i in range(anchor_pos+2, p.size()+1): # skipping the overlapping resiude on the scaffold
        if 'Nterm' in p.residue(i).name():
            new_p.append_residue_by_jump(p.residue(i),new_p.size())
        else:
            new_p.append_residue_by_bond(p.residue(i))
    return new_p

def insert_c_cap(p, anchor_pos, cap):
    # inserting ccap to the head side of the gap in pdb (closer to pdb's nterm)
    # assuming ncap has been inserted before ccap!
    p_anchor_coord = get_anchor_coordinates_from_pose(p, [anchor_pos])
    align_cap_pose_to_anchor_coords(cap, p_anchor_coord, ncap=False)
    new_p = pyrosetta.rosetta.core.pose.Pose()
    new_p.append_residue_by_jump(p.residue(1), 1)
    for i in range(2,anchor_pos): # skipping the overlapping resiude on the scaffold
        new_p.append_residue_by_bond(p.residue(i))
    for i in range(2,cap.size()):
        new_p.append_residue_by_bond(cap.residue(i))
    new_p.append_residue_by_jump(p.residue(anchor_pos+1), anchor_pos+cap.size()-3)
    for i in range(anchor_pos+2, p.size()+1): 
        if 'Nterm' in p.residue(i).name():
            new_p.append_residue_by_jump(p.residue(i),new_p.size())
        else:        
            new_p.append_residue_by_bond(p.residue(i))
    return new_p

def read_torsion_file(torsion_file):
    phipsi = []
    if torsion_file == '':
        return phipsi
    elif 'angle' in torsion_file:
        readlist = [torsion_file]
    else:
        readlist = open(torsion_file,'r').readlines()
    for m in readlist:
        pp = []
        c = 0
        lines = open(m.strip(),'r').readlines()
        for line in lines:
            angles = [float(x) for x in line.split()[1:3]]
            if (c == 0 or c == len(lines)-1) and 0 in angles:  # hacky check to skip the terminal residues
                continue
            pp.append(angles)
            c += 1
        phipsi.append([pp, m.strip()])
    return phipsi


def add_caps_n_filter_direction( pose,
                                sf,
                                sf_farep,
                                sf_farep_cutoff,
                                  c_cap_phipsi_file,
                                  n_cap_phipsi_file,
                                h1_len = 20, 
                                h2_len = 20, 
                                cap_search_range = 4,
                                num_res_for_helix_center_com = 4,
                                min_cos_angle = 0.8,
                                min_helix_height_diff = 4.0,
                                max_helix_height_diff = 7.0):

    '''

    CAUTION:
        in this version, ncap is added to beginning of h1, ccap added to end of h2
    
    # check 1st-4th residues from each helix terminus for vector angles
    
    # also make sure the ccap is 'higher' than ncap so loop won't clash into ncap
    
    
    # test values
    cap_dir = 'capping_files'  # obsolete
    h1_len, h2_len = 20, 20 ## this should be determined beforehand!!
    cap_search_range = 4
    num_res_for_helix_center_com = 4
    min_cos_angle = 0.8 ## min cos value of vec(helix_center->capping res) and vec(helix_center-> Zaxis) ... to be optimized    
    
    '''


    sf_cap = sf.clone()
    sf_cap.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.2) # soft rep to encourage PRO packing for capping residues
    #if aacomp_cap_pro_file != "-1":
    #    sf_cap.set_weight(pyrosetta.rosetta.core.scoring.aa_composition, 1) # turn on aacomp


    # ccap (end of 1st helix)
    #c_cap_phipsi = read_torsion_file('{}{}/{}/angle_ccap_ank1_4aa_SKGA'.format(workdir, output_dir, cap_dir))
    c_cap_phipsi = read_torsion_file(c_cap_phipsi_file)
    #print(c_cap_phipsi)

    pose_ccap_list = []
    for i in range(cap_search_range):
        pose_ccap = pose.clone()
        # ccap to be put at the end of h2
        anchor_phipsi = [[pose_ccap.phi(h1_len+h2_len-1-i),pose_ccap.psi(h1_len+h2_len-1-i)]]
        ccap = generate_cap(anchor_phipsi+c_cap_phipsi[0][0], sf_cap, aa='A', insert_Pro=False, insert_GLy=True, aacomp_cap_pro_file="-1")
        pose_ccap.delete_residue_range_slow(h1_len+h2_len-1-i, h1_len+h2_len)
        pose_ccap = insert_c_cap(pose_ccap, h1_len+h2_len-2-i, ccap)
        #pose_ccap.dump_pdb('{}{}/{}/{}_ccap_-{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, i))

        helix_center_reslist = [h1_len+h2_len-i+1-x for x in range(num_res_for_helix_center_com)]   
        #print(helix_center_reslist)
        com_helix_center = np.array(get_com(pose_ccap, helix_center_reslist))
        #print(com_helix_center)

        cap_reslist = [h1_len+h2_len-i+2]
        com_cap = np.array(get_com(pose_ccap, cap_reslist))    
        vec_hc_cap = com_cap - com_helix_center
        # simple hack of projection of vec_hc_cap on XY plane (i.e. perpendicular to Z-axis)
        vec_hc_cap[2] = 0
        unit_vec_hc_cap = vec_hc_cap / np.linalg.norm(vec_hc_cap)
        vec_hc_z = np.array([0-com_helix_center[0], 0-com_helix_center[1], 0])
        unit_vec_hc_z = vec_hc_z / np.linalg.norm(vec_hc_z)

        cos_angle = np.dot(unit_vec_hc_cap, unit_vec_hc_z)   

        #print('i, ccap_cos_angle: ', i, cos_angle)

        if cos_angle >= min_cos_angle:
            pose_ccap_list.append([pose_ccap, cos_angle, i])


    #print(pose_ccap_list)

    if len(pose_ccap_list) == 0:
        #print('Warning: no ccap satisfies the min_cos_angle cutoff, stop checking ncaps ...') 
        return []

    else:
        # ncap (start of 2nd helix, after ccap insertion)
        #n_cap_phipsi = read_torsion_file('{}{}/{}/angle_ncap_ank1_4aa_RTPL'.format(workdir, output_dir, cap_dir))
        n_cap_phipsi = read_torsion_file(n_cap_phipsi_file)
        #print(n_cap_phipsi)


        pose_ncap_list = []
        for c_id in range(len(pose_ccap_list)):
            for i in range(cap_search_range):
                pose_ncap = pose_ccap_list[c_id][0].clone()
                
                # use the 2nd res phipsi in 2nd helix as anchor as the terminal res has only psi
                anchor_phipsi = [[pose_ncap.phi(2+i),pose_ncap.psi(2+i)]]   
                ncap = generate_cap(n_cap_phipsi[0][0]+anchor_phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=False, aacomp_cap_pro_file="-1")
                pose_ncap.delete_residue_range_slow(1, 1+i)
                #pose_ncap = insert_n_cap(pose_ncap, pose_ccap_h1_len, ncap)  
                pose_ncap = prepend_n_cap(pose_ncap, ncap)
                #pose_ncap.dump_pdb('{}{}/{}/{}_ccap_-{}_ncap_+{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, c_id, i))


                helix_center_reslist = [i+1+x for x in range(num_res_for_helix_center_com)]  
                #print(helix_center_reslist)
                com_helix_center = np.array(get_com(pose_ncap, helix_center_reslist))
                #print(com_helix_center)

                cap_reslist = [1]
                com_cap = np.array(get_com(pose_ncap, cap_reslist))    
                vec_hc_cap = com_cap - com_helix_center
                # simple hack of projection of vec_hc_cap on XY plane (i.e. perpendicular to Z-axis)
                vec_hc_cap[2] = 0
                unit_vec_hc_cap = vec_hc_cap / np.linalg.norm(vec_hc_cap)
                vec_hc_z = np.array([0-com_helix_center[0], 0-com_helix_center[1], 0])
                unit_vec_hc_z = vec_hc_z / np.linalg.norm(vec_hc_z)

                cos_angle = np.dot(unit_vec_hc_cap, unit_vec_hc_z)   
                #print('i, ncap_cos_angle: ',i, cos_angle)

                # check the height of ccap and ncap
                #      ncap_size - ncap_trim_res_num + original_h1 + original_h2 - ccap_trim_res_num + 2 helical res before capping res
                ccap_z = pose_ncap.residue(len(n_cap_phipsi[0][0])-i-2+h1_len+h2_len-pose_ccap_list[c_id][2]+2).xyz('CA')[2]  
                ncap_z = pose_ncap.residue(3).xyz('CA')[2]
                #print('ccap resid: {} ncap resid: {}'.format(h1_len+h2_len-pose_ccap_list[c_id][2]+2, 1))
                if np.abs(ccap_z - ncap_z) < min_helix_height_diff or np.abs(ccap_z - ncap_z) > max_helix_height_diff:
                    continue
                
                if cos_angle >= min_cos_angle:
                    farep_score = sf_farep(pose_ncap)
                    #print(farep_score)
                    if farep_score <= sf_farep_cutoff:
                        new_repeat_len = len(n_cap_phipsi[0][0])-i-1+h1_len+h2_len-pose_ccap_list[c_id][2]+len(c_cap_phipsi[0][0])-2
                        # add ncap to the 2nd repeat to prep for propagation
                        pose_ncap.delete_residue_range_slow(new_repeat_len+1, new_repeat_len+1+i)
                        pose_ncap = insert_n_cap(pose_ncap, new_repeat_len, ncap)
                        pose_ncap_list.append([pose_ncap, pose_ccap_list[c_id][1], pose_ccap_list[c_id][2], cos_angle, i, new_repeat_len])

        if len(pose_ncap_list) == 0:
            #print('Warning: no ncap satisfies the min_cos_angle cutoff, no good capped scaffold for loop sampling ...') 
            return []
        else:
            #for n_id in range(len(pose_ncap_list)):
                #pose_ncap_list[n_id][0].dump_pdb('{}{}/{}/{}_ccap_-{}_ncap_+{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, \
                #                                                            pose_ncap_list[n_id][2], pose_ncap_list[n_id][4]))
                #print('ccap_id: %d  ccap_cos: %.3f  ncap_id: %d  ncap_cos: %.3f' % (pose_ncap_list[n_id][2], pose_ncap_list[n_id][1], \
                #                                                                    pose_ncap_list[n_id][4], pose_ncap_list[n_id][3]))

            #print('best ccap: {}'.format(sorted(pose_ccap_list, key=lambda x:x[1], reverse=True)[0][2]))    
            #print('best ncap: {}'.format(sorted(pose_ncap_list, key=lambda x:x[1], reverse=True)[0][4]))   
            
            #return [x[0] for x in pose_ncap_list]
            return sorted(pose_ncap_list, key=lambda x:x[1]+x[3], reverse=True)
    

def add_caps_n_filter_direction_w_control( pose,
                                sf,
                                sf_farep,
                                sf_farep_cutoff,
                                  c_cap_phipsi_file,
                                  n_cap_phipsi_file,
                                h1_len = 20, 
                                h2_len = 20, 
                                cap_search_range = 4,
                                num_res_for_helix_center_com = 4,
                                min_cos_angle = 0.8,
                                ccap_min_cos_angle = 10,
                                ccap_direction = 0, # 0: no direction preference, 1: left, -1: right
                                ncap_min_cos_angle = 10,
                                ncap_direction = 0, # 0: no direction preference, 1: left, -1: right
                                min_helix_height_diff = 4.0,
                                max_helix_height_diff = 7.0):

    '''
    Difference from add_caps_n_filter_direction:
        1. min_cos_angle is specified indivdually for ccap and ncap
        2. allowing specifying of cap direction with respect to the Z-axis+cap_com plane 
            (so I can select only left, or only right pointing caps)


    CAUTION:
        in this version, ncap is added to beginning of h1, ccap added to end of h2
    
    # check 1st-4th residues from each helix terminus for vector angles
    
    # also make sure the ccap is 'higher' than ncap so loop won't clash into ncap
    
    
    # test values
    cap_dir = 'capping_files'  # obsolete
    h1_len, h2_len = 20, 20 ## this should be determined beforehand!!
    cap_search_range = 4
    num_res_for_helix_center_com = 4
    min_cos_angle = 0.8 ## min cos value of vec(helix_center->capping res) and vec(helix_center-> Zaxis) ... to be optimized    
    
    '''


    sf_cap = sf.clone()
    sf_cap.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.2) # soft rep to encourage PRO packing for capping residues
    #if aacomp_cap_pro_file != "-1":
    #    sf_cap.set_weight(pyrosetta.rosetta.core.scoring.aa_composition, 1) # turn on aacomp


    # ccap (end of 1st helix)
    #c_cap_phipsi = read_torsion_file('{}{}/{}/angle_ccap_ank1_4aa_SKGA'.format(workdir, output_dir, cap_dir))
    c_cap_phipsi = read_torsion_file(c_cap_phipsi_file)
    #print(c_cap_phipsi)

    pose_ccap_list = []
    for i in range(cap_search_range):
        pose_ccap = pose.clone()
        # ccap to be put at the end of h2
        anchor_phipsi = [[pose_ccap.phi(h1_len+h2_len-1-i),pose_ccap.psi(h1_len+h2_len-1-i)]]
        ccap = generate_cap(anchor_phipsi+c_cap_phipsi[0][0], sf_cap, aa='A', insert_Pro=False, insert_GLy=True, aacomp_cap_pro_file="-1")
        pose_ccap.delete_residue_range_slow(h1_len+h2_len-1-i, h1_len+h2_len)
        pose_ccap = insert_c_cap(pose_ccap, h1_len+h2_len-2-i, ccap)
        #pose_ccap.dump_pdb('{}{}/{}/{}_ccap_-{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, i))

        helix_center_reslist = [h1_len+h2_len-i+1-x for x in range(num_res_for_helix_center_com)]   
        #print(helix_center_reslist)
        com_helix_center = np.array(get_com(pose_ccap, helix_center_reslist))
        #print(com_helix_center)

        cap_reslist = [h1_len+h2_len-i+2]
        com_cap = np.array(get_com(pose_ccap, cap_reslist))    
        vec_hc_cap = com_cap - com_helix_center
        # simple hack of projection of vec_hc_cap on XY plane (i.e. perpendicular to Z-axis)
        vec_hc_cap[2] = 0
        unit_vec_hc_cap = vec_hc_cap / np.linalg.norm(vec_hc_cap)
        vec_hc_z = np.array([0-com_helix_center[0], 0-com_helix_center[1], 0])
        unit_vec_hc_z = vec_hc_z / np.linalg.norm(vec_hc_z)

        cos_angle = np.dot(unit_vec_hc_cap, unit_vec_hc_z)   

        #print('i, ccap_cos_angle: ', i, cos_angle)

        if ccap_min_cos_angle > 1:
            ccap_min_cos_angle = min_cos_angle

        if cos_angle >= ccap_min_cos_angle:

            # now check cap direction relative to the Z-axis+cap_com plane
            if ccap_direction != 0:
                # computing determinant of the matrix from stacking 3 vectors:
                #    com_helix_center, com_helix_center projected on XY plane, com_cap
                #    if det * cap_direction >= 0, direction is correct
                com_helix_center_xy = np.copy(com_helix_center)
                if com_helix_center[2] != 0:
                    com_helix_center_xy[2] = 0
                else:
                    com_helix_center_xy[2] = com_helix_center[2] - 1 # incase com_helix_center happens to be in XY plane
                m = np.stack((com_helix_center, com_helix_center_xy, com_cap))
                if np.linalg.det(m) * ccap_direction >= 0:
                    pose_ccap_list.append([pose_ccap, cos_angle, i])
            else:
                pose_ccap_list.append([pose_ccap, cos_angle, i])


    #print(pose_ccap_list)

    if len(pose_ccap_list) == 0:
        #print('Warning: no ccap satisfies the min_cos_angle cutoff, stop checking ncaps ...') 
        return []

    else:
        # ncap (start of 2nd helix, after ccap insertion)
        #n_cap_phipsi = read_torsion_file('{}{}/{}/angle_ncap_ank1_4aa_RTPL'.format(workdir, output_dir, cap_dir))
        n_cap_phipsi = read_torsion_file(n_cap_phipsi_file)
        #print(n_cap_phipsi)


        pose_ncap_list = []
        for c_id in range(len(pose_ccap_list)):
            for i in range(cap_search_range):
                pose_ncap = pose_ccap_list[c_id][0].clone()
                
                # use the 2nd res phipsi in 2nd helix as anchor as the terminal res has only psi
                anchor_phipsi = [[pose_ncap.phi(2+i),pose_ncap.psi(2+i)]]   
                ncap = generate_cap(n_cap_phipsi[0][0]+anchor_phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=False, aacomp_cap_pro_file="-1")
                pose_ncap.delete_residue_range_slow(1, 1+i)
                #pose_ncap = insert_n_cap(pose_ncap, pose_ccap_h1_len, ncap)  
                pose_ncap = prepend_n_cap(pose_ncap, ncap)
                #pose_ncap.dump_pdb('{}{}/{}/{}_ccap_-{}_ncap_+{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, c_id, i))


                helix_center_reslist = [i+1+x for x in range(num_res_for_helix_center_com)]  
                #print(helix_center_reslist)
                com_helix_center = np.array(get_com(pose_ncap, helix_center_reslist))
                #print(com_helix_center)

                cap_reslist = [1]
                com_cap = np.array(get_com(pose_ncap, cap_reslist))    
                vec_hc_cap = com_cap - com_helix_center
                # simple hack of projection of vec_hc_cap on XY plane (i.e. perpendicular to Z-axis)
                vec_hc_cap[2] = 0
                unit_vec_hc_cap = vec_hc_cap / np.linalg.norm(vec_hc_cap)
                vec_hc_z = np.array([0-com_helix_center[0], 0-com_helix_center[1], 0])
                unit_vec_hc_z = vec_hc_z / np.linalg.norm(vec_hc_z)

                cos_angle = np.dot(unit_vec_hc_cap, unit_vec_hc_z)   
                #print('i, ncap_cos_angle: ',i, cos_angle)

                # check the height of ccap and ncap
                #      ncap_size - ncap_trim_res_num + original_h1 + original_h2 - ccap_trim_res_num + 2 helical res before capping res
                ccap_z = pose_ncap.residue(len(n_cap_phipsi[0][0])-i-2+h1_len+h2_len-pose_ccap_list[c_id][2]+2).xyz('CA')[2]  
                ncap_z = pose_ncap.residue(3).xyz('CA')[2]
                #print('ccap resid: {} ncap resid: {}'.format(h1_len+h2_len-pose_ccap_list[c_id][2]+2, 1))
                if np.abs(ccap_z - ncap_z) < min_helix_height_diff or np.abs(ccap_z - ncap_z) > max_helix_height_diff:
                    continue
                
                if ncap_min_cos_angle > 1:
                    ncap_min_cos_angle = min_cos_angle

                if cos_angle >= ncap_min_cos_angle:

                    #TODO: copy ccap_direction code here for ncap_direction filtering

                    farep_score = sf_farep(pose_ncap)
                    #print(farep_score)
                    if farep_score <= sf_farep_cutoff:
                        new_repeat_len = len(n_cap_phipsi[0][0])-i-1+h1_len+h2_len-pose_ccap_list[c_id][2]+len(c_cap_phipsi[0][0])-2
                        # add ncap to the 2nd repeat to prep for propagation
                        pose_ncap.delete_residue_range_slow(new_repeat_len+1, new_repeat_len+1+i)
                        pose_ncap = insert_n_cap(pose_ncap, new_repeat_len, ncap)
                        pose_ncap_list.append([pose_ncap, pose_ccap_list[c_id][1], pose_ccap_list[c_id][2], cos_angle, i, new_repeat_len])

        if len(pose_ncap_list) == 0:
            #print('Warning: no ncap satisfies the min_cos_angle cutoff, no good capped scaffold for loop sampling ...') 
            return []
        else:
            #for n_id in range(len(pose_ncap_list)):
                #pose_ncap_list[n_id][0].dump_pdb('{}{}/{}/{}_ccap_-{}_ncap_+{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, \
                #                                                            pose_ncap_list[n_id][2], pose_ncap_list[n_id][4]))
                #print('ccap_id: %d  ccap_cos: %.3f  ncap_id: %d  ncap_cos: %.3f' % (pose_ncap_list[n_id][2], pose_ncap_list[n_id][1], \
                #                                                                    pose_ncap_list[n_id][4], pose_ncap_list[n_id][3]))

            #print('best ccap: {}'.format(sorted(pose_ccap_list, key=lambda x:x[1], reverse=True)[0][2]))    
            #print('best ncap: {}'.format(sorted(pose_ncap_list, key=lambda x:x[1], reverse=True)[0][4]))   
            
            #return [x[0] for x in pose_ncap_list]
            return sorted(pose_ncap_list, key=lambda x:x[1]+x[3], reverse=True)


def add_caps_n_filter_direction_for_scaffold( pose,
                                sf,
                                sf_farep,
                                sf_farep_cutoff,
                                  c_cap_phipsi_file,
                                  n_cap_phipsi_file,
                                h1_len = 20, # ccap helix
                                #h2_len = 20, # obsolete
                                cap_search_range = 4,
                                num_res_for_helix_center_com = 4,
                                min_cos_angle_scaffold = 0.5,
                                min_helix_height_diff = 4.0,
                                max_helix_height_diff = 7.0):

    '''
    
    # check 1st-4th residues from each helix terminus for vector angles
    
    # also make sure the ccap is 'higher' than ncap so loop won't clash into ncap
    
    
    # test values
    cap_dir = 'capping_files'  # obsolete
    h1_len, h2_len = 20, 20 ## this should be determined beforehand!!
    cap_search_range = 4
    num_res_for_helix_center_com = 4
    min_cos_angle_scaffold = 0.5 ## min cos value of vec(helix_center->capping res) and vec(helix_center-> Zaxis) ... to be optimized 
                        this is for the capping of the longloopless scaffolds!   
    
    '''


    sf_cap = sf.clone()
    sf_cap.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.2) # soft rep to encourage PRO packing for capping residues
    #if aacomp_cap_pro_file != "-1":
    #    sf_cap.set_weight(pyrosetta.rosetta.core.scoring.aa_composition, 1) # turn on aacomp


    # ccap (end of 1st helix)
    #c_cap_phipsi = read_torsion_file('{}{}/{}/angle_ccap_ank1_4aa_SKGA'.format(workdir, output_dir, cap_dir))
    c_cap_phipsi = read_torsion_file(c_cap_phipsi_file)
    #print(c_cap_phipsi)

    pose_ccap_list = []
    for i in range(cap_search_range):
        pose_ccap = pose.clone()
        anchor_phipsi = [[pose_ccap.phi(h1_len-1-i),pose_ccap.psi(h1_len-1-i)]]
        ccap = generate_cap(anchor_phipsi+c_cap_phipsi[0][0], sf_cap, aa='A', insert_Pro=False, insert_GLy=True, aacomp_cap_pro_file="-1")
        pose_ccap.delete_residue_range_slow(h1_len-1-i, h1_len)
        pose_ccap = insert_c_cap(pose_ccap, h1_len-2-i, ccap)
        #pose_ccap.dump_pdb('{}{}/{}/{}_ccap_-{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, i))

        ccap_helix_center_reslist = [h1_len-i+1-x for x in range(num_res_for_helix_center_com)]   
        #print(ccap_helix_center_reslist)
        ccap_com_helix_center = np.array(get_com(pose_ccap, ccap_helix_center_reslist))
        #print(ccap_com_helix_center)

        pose_ccap_h1_len = h1_len - i + 2
        ncap_helix_center_reslist = [pose_ccap_h1_len+1+x for x in range(num_res_for_helix_center_com)]
        ncap_com_helix_center = np.array(get_com(pose_ccap, ncap_helix_center_reslist))

        ccap_reslist = [h1_len-i+2]
        com_ccap = np.array(get_com(pose_ccap, ccap_reslist))    
        vec_chc_ccap = com_ccap - ccap_com_helix_center
        vec_chc_nhc = ncap_com_helix_center - ccap_com_helix_center
        # simple hack of projection of vec_hc_cap on XY plane (i.e. perpendicular to Z-axis)
        vec_chc_ccap[2] = 0
        vec_chc_nhc[2] = 0
        unit_vec_chc_ccap = vec_chc_ccap / np.linalg.norm(vec_chc_ccap)
        unit_vec_chc_nhc = vec_chc_nhc / np.linalg.norm(vec_chc_nhc)

        cos_angle = np.dot(unit_vec_chc_ccap, unit_vec_chc_nhc)   

        #print(i, cos_angle)

        if cos_angle >= min_cos_angle_scaffold:
            pose_ccap_list.append([pose_ccap, cos_angle, i])


    #print(pose_ccap_list)

    if len(pose_ccap_list) == 0:
        #print('Warning: no ccap satisfies the min_cos_angle cutoff, stop checking ncaps ...') 
        return []

    else:
        # ncap (start of 2nd helix, after ccap insertion)
        #n_cap_phipsi = read_torsion_file('{}{}/{}/angle_ncap_ank1_4aa_RTPL'.format(workdir, output_dir, cap_dir))
        n_cap_phipsi = read_torsion_file(n_cap_phipsi_file)
        #print(n_cap_phipsi)


        pose_ncap_list = []
        for c_id in range(len(pose_ccap_list)):
            for i in range(cap_search_range):
                pose_ncap = pose_ccap_list[c_id][0].clone()
                pose_ccap_h1_len = h1_len - pose_ccap_list[c_id][2] + 2
                
                # use the 2nd res phipsi in 2nd helix as anchor as the terminal res has only psi
                anchor_phipsi = [[pose_ncap.phi(pose_ccap_h1_len+2+i),pose_ncap.psi(pose_ccap_h1_len+2+i)]]   
                ncap = generate_cap(n_cap_phipsi[0][0]+anchor_phipsi, sf_cap, aa='A', insert_Pro=True, insert_GLy=False, aacomp_cap_pro_file="-1")
                pose_ncap.delete_residue_range_slow(pose_ccap_h1_len+1, pose_ccap_h1_len+1+i)
                pose_ncap = insert_n_cap(pose_ncap, pose_ccap_h1_len, ncap)  
                #pose_ncap.dump_pdb('{}{}/{}/{}_ccap_-{}_ncap_+{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, c_id, i))


                ccap_helix_center_reslist = [h1_len-pose_ccap_list[c_id][2]+1-x for x in range(num_res_for_helix_center_com)]   
                #print(ccap_helix_center_reslist)
                ccap_com_helix_center = np.array(get_com(pose_ncap, ccap_helix_center_reslist))
                #print(ccap_com_helix_center)                


                ncap_helix_center_reslist = [pose_ccap_h1_len+i+1+x for x in range(num_res_for_helix_center_com)]  
                #print(ncap_helix_center_reslist)
                ncap_com_helix_center = np.array(get_com(pose_ncap, ncap_helix_center_reslist))
                #print(ncap_com_helix_center)

                ncap_reslist = [pose_ccap_h1_len+1]
                com_ncap = np.array(get_com(pose_ncap, ncap_reslist))    
                vec_nhc_ncap = com_ncap - ncap_com_helix_center
                vec_nhc_chc = ccap_com_helix_center - ncap_com_helix_center
                # simple hack of projection of vec_hc_cap on XY plane (i.e. perpendicular to Z-axis)
                vec_nhc_ncap[2] = 0
                vec_nhc_chc[2] = 0

                unit_vec_nhc_ncap = vec_nhc_ncap / np.linalg.norm(vec_nhc_ncap)
                unit_vec_nhc_chc = vec_nhc_chc / np.linalg.norm(vec_nhc_chc)
                cos_angle = np.dot(unit_vec_nhc_ncap, unit_vec_nhc_chc)   

                # check the height of ccap and ncap (not really necessary for the scaffold here)
                ccap_z = pose_ncap.residue(pose_ccap_h1_len).xyz('CA')[2]
                ncap_z = pose_ncap.residue(pose_ccap_h1_len+1).xyz('CA')[2]
                if np.abs(ccap_z - ncap_z) < min_helix_height_diff or np.abs(ccap_z - ncap_z) > max_helix_height_diff:
                    continue
                
                #print(i, cos_angle)

                # filter ccap-ncap relative direction (they should point to each other)
                if cos_angle >= min_cos_angle_scaffold:
                    farep_score = sf_farep(pose_ncap)
                    if farep_score <= sf_farep_cutoff:
                        pose_ncap_list.append([pose_ncap, pose_ccap_list[c_id][1], pose_ccap_list[c_id][2], cos_angle, i])

        if len(pose_ncap_list) == 0:
            #print('Warning: no ncap satisfies the min_cos_angle cutoff, no good capped scaffold for loop sampling ...') 
            return []
        else:
            #for n_id in range(len(pose_ncap_list)):
                #pose_ncap_list[n_id][0].dump_pdb('{}{}/{}/{}_ccap_-{}_ncap_+{}.pdb'.format(workdir, output_dir, cap_dir, input_pre, \
                #                                                            pose_ncap_list[n_id][2], pose_ncap_list[n_id][4]))
                #print('ccap_id: %d  ccap_cos: %.3f  ncap_id: %d  ncap_cos: %.3f' % (pose_ncap_list[n_id][2], pose_ncap_list[n_id][1], \
                #                                                                    pose_ncap_list[n_id][4], pose_ncap_list[n_id][3]))

            #print('best ccap: {}'.format(sorted(pose_ccap_list, key=lambda x:x[1], reverse=True)[0][2]))    
            #print('best ncap: {}'.format(sorted(pose_ncap_list, key=lambda x:x[1], reverse=True)[0][4]))   
            
            #return [x[0] for x in pose_ncap_list]
            return sorted(pose_ncap_list, key=lambda x:x[1]+x[3], reverse=True)

def compute_motifscore_for_residue_pair(pose, res1, res2, dist_cutoff=12, 
                                        dssp_obj=None, motif_hash_man=None, max_motif_per_res=3.0):
    '''
        modified from core/scoring/methods/CenPairMotifEnergy.cc
                    protocol/simple_filters/SSElementMotifContactFilter::get_SSelements_in_contact
    '''
    
    tree = pose.fold_tree()
    
    # do not score terminal residues (lack bb torsion) or self pairig or jump point in foldtree
    if res1 in [1, pose.size()] or res2 in [1, pose.size()] or res1 == res2 or \
        tree.is_jump_point(res1) or tree.is_jump_point(res2):
        return 0
    
    if dssp_obj == None:
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        dssp_obj.dssp_reduced()
        #dssp = dssp_obj.get_dssp_secstruct()
    
    if motif_hash_man == None:
        motif_hash_man = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()
           
    dist = pose.residue(res1).xyz('CA').distance(pose.residue(res2).xyz('CA'))
    # skip motif searching if residues too far away from each other
    if dist > dist_cutoff:
        return 0
    
    ss1 = dssp_obj.get_dssp_secstruct(res1)
    aa1 = pose.residue(res1).name1()
    bb_stub1 = pyrosetta.rosetta.core.pose.motif.get_backbone_reference_frame(pose, res1)

    ss2 = dssp_obj.get_dssp_secstruct(res2)
    aa2 = pose.residue(res2).name1()    
    bb_stub2 = pyrosetta.rosetta.core.pose.motif.get_backbone_reference_frame(pose, res2)

    #print(ss1,aa1,ss2,aa2,dist)
    
    # Xbb = bb_stub1.inverse() * bb_stub2
    # numeric/xyzTransform.hh: line 127:
    # friend Transform operator *( Transform const & a, Transform const & b ){ return Transform( a.R*b.R, a.R*b.t + a.t ); }
    bb_stub1_inv = bb_stub1.inverse()
    Xbb = pyrosetta.rosetta.numeric.xyzTransform_double_t( bb_stub1_inv.R*bb_stub2.R, bb_stub1_inv.R*bb_stub2.t + bb_stub1_inv.t )
    
    xs_bb_fxn1 = motif_hash_man.get_xform_score_BB_BB(ss1,ss2,aa1,aa2)
    xs_bb_fxn2 = motif_hash_man.get_xform_score_BB_BB(ss2,ss1,aa2,aa1)
    
    try:        
        score1 = xs_bb_fxn1.score_of_bin(Xbb)
    except AttributeError:
        score1 = 0    
    try:        
        score2 = xs_bb_fxn2.score_of_bin(Xbb.inverse())
    except AttributeError:
        score2 = 0
        
    tmpScore = min(score1+score2, max_motif_per_res)
        
    return tmpScore


def compute_motifscore_between_poses(pose1, pose2, pose1_reslist=[], pose2_reslist=[], motif_dist_cutoff=10.0, 
                                     dssp_obj=None, mman_=None):
    pose = pose1.clone()
    pose.append_pose_by_jump(pose2, pose1.size())
    
    sf_motif = sf.clone()
    for st in sf_motif.get_nonzero_weighted_scoretypes():
        sf_motif.set_weight(st, 0)
    sf_motif.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cen_pair_motifs, 1)
    #print('sf: ', sf(pose))
    #print('sf_motif: ', sf_motif(pose))
    
    if dssp_obj == None:
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        dssp_obj.dssp_reduced()
    if mman_ == None:
        mman_ = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()
    
    if len(pose1_reslist) == 0:
        pose1_reslist = range(1, pose1.size()+1)
    if len(pose2_reslist) == 0:
        pose2_reslist = range(1, pose2.size()+1)
    pose2_reslist = [x+pose1.size() for x in pose2_reslist]
        
    total_motifscore = 0
    for res1 in pose1_reslist:
        for res2 in pose2_reslist:
            motifscore = compute_motifscore_for_residue_pair(pose, res1, res2, dist_cutoff=motif_dist_cutoff, dssp_obj=dssp_obj, motif_hash_man=mman_)
            #print(res1, res2, motifscore)
            total_motifscore += motifscore
            
    return -1*total_motifscore / pose.size()

def compute_motifscore_within_pose(pose, reslist1=[], reslist2=[], motif_dist_cutoff=10.0, dssp_obj=None, mman_=None):   
    if dssp_obj == None:
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        dssp_obj.dssp_reduced()
    if mman_ == None:
        mman_ = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()
    
    if len(reslist1) == 0:
        reslist1 = range(1, pose.size()+1)
    if len(reslist2) == 0:
        reslist2 = range(1, pose.size()+1)    
    
    total_motifscore = 0
    for res1 in reslist1:
        for res2 in reslist2:
            if res1 == res2:
                continue
            motifscore = compute_motifscore_for_residue_pair(pose, res1, res2, dist_cutoff=motif_dist_cutoff, dssp_obj=dssp_obj, motif_hash_man=mman_)
            #print(res1, res2, motifscore)
            total_motifscore += motifscore
            
            #if motifscore > 0:
            #    print(res1, res2, motifscore)
            
    return -1*total_motifscore / pose.size()

def find_loopless_dhr_chains(pose, breakcutoff=1.5):
    chains = []
    start = 1
    curr = 1   
    for i in range(2, pose.size()+1):
        if pose.residue(i).xyz('N').distance(pose.residue(curr).xyz('C')) > breakcutoff:
            chains.append([start, i-1])
            start = i
        curr = i
    if start != pose.size():
        chains.append([start, pose.size()])
    return chains
    
    
def compute_ss_degree(pose, helices, helix_id, motifscore_cutoff = -0.01, dssp_obj=None, mman_=None, debug=False):
    if dssp_obj == None:
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        dssp_obj.dssp_reduced()
    if mman_ == None:
        mman_ = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()    
    
    ss_degree = 0
    this_helix = helices[helix_id]
    ss_degree_list = []
    for i, h in enumerate(helices):
        if h == this_helix:
            continue
        motifscore = compute_motifscore_within_pose(pose, reslist1=range(this_helix[0],this_helix[1]+1), 
                                                    reslist2=range(h[0],h[1]+1) , dssp_obj=dssp_obj, mman_=mman_)
        if motifscore <= motifscore_cutoff:
            ss_degree += 1
            ss_degree_list.append([i,motifscore])
            #if _DEBUG:
            #    print(motifscore)
    if debug:
        return ss_degree_list, ss_degree
    else:
        return ss_degree


def compute_ss_degree_quick_n_dirty(pose, helices, helix_id, motifscore_cutoff = -0.01, dssp_obj=None, mman_=None, debug=False):
    '''
        hacky implementation to speed up
    '''
    if dssp_obj == None:
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        dssp_obj.dssp_reduced()
    if mman_ == None:
        mman_ = pyrosetta.rosetta.core.scoring.motif.MotifHashManager.get_instance()    
    
    ss_degree = 0
    this_helix = helices[helix_id]

    com_list = [np.array(pose.residue(h[0]).xyz('CA') + pose.residue(h[1]).xyz('CA'))/2 for h in helices]

    ss_degree_list = []
    for i, h in enumerate(helices):
        if h == this_helix:
            continue
        if np.linalg.norm(com_list[helix_id] - com_list[i]) > 15:
            continue
        motifscore = compute_motifscore_within_pose(pose, reslist1=range(this_helix[0],this_helix[1]+1), 
                                                    reslist2=range(h[0],h[1]+1) , dssp_obj=dssp_obj, mman_=mman_)
        if motifscore <= motifscore_cutoff:
            ss_degree += 1
            ss_degree_list.append([i,motifscore])
            #if _DEBUG:
            #    print(motifscore)
    if debug:
        return ss_degree_list, ss_degree
    else:
        return ss_degree


def tilt_inner_row_helices(pose_prop_cap, repeat_len, num_repeat=5, num_inner_helix_tilt_direction=2, num_inner_helix_tilt_degree=5, min_tilt_degree=0):
 

    '''         
    #  num_inner_helix_tilt_direction (int, >=0): number of different tilt directions to try
            0: no tilting, 
            1: tilt only to the right centerofmass...
            2: tilt to both right and left centerofmass...
            >3: evenly divided the rotation angel from right com to left com by this number and tilt along these directions
    #  num_inner_helix_tilt_degree (int, >=0): number of different degree of tiling of inner row helix to try
            0: no tilting, 
            1: no tilting, PLUS use ncap com as rotation axis, tilt ccap toward centerofmass of the neighbor helices at bottom
            2: 0,1 and one more that tilt half way through towards the centerofmass...
            >2: evenly divided 1 by this number and tilt

    '''
    if min_tilt_degree == 0:
        output_pose_list = [[pose_prop_cap,0,0]] # [pose, tilt_direction_index, tilt_degree_index]
    else:
        output_pose_list = []

    if num_inner_helix_tilt_direction == 0 or num_inner_helix_tilt_degree == 0:
        return output_pose_list

    helices = find_loopless_dhr_chains(pose_prop_cap)

    # get h3ncap com
    ncap_h3_reslist = [helices[2][0]+i for i in range(4)]
    ncap_h3_com = np.array(get_com(pose_prop_cap, ncap_h3_reslist))
    #if _DEBUG:
    #    print('ncap_h3_com: ', ncap_h3_com)

    # get the center of h3,h5's ccaps com and h4,h6's ncaps coms
    ccap_h3_reslist = [helices[2][1]-i for i in range(4)]
    ccap_h3_com = np.array(get_com(pose_prop_cap, ccap_h3_reslist))

    ccap_h5_reslist = [helices[4][1]-i for i in range(4)]
    ccap_h5_com = np.array(get_com(pose_prop_cap, ccap_h5_reslist))

    ncap_h4_reslist = [helices[3][0]+i for i in range(4)]
    ncap_h4_com = np.array(get_com(pose_prop_cap, ncap_h4_reslist))

    ncap_h6_reslist = [helices[5][0]+i for i in range(4)]
    ncap_h6_com = np.array(get_com(pose_prop_cap, ncap_h6_reslist))

    caps_com = (ccap_h3_com + ccap_h5_com + ncap_h4_com + ncap_h6_com)/4
    #if _DEBUG:
    #    print('caps_com: ', caps_com)

    # get the center of h1,h3's ccaps com and h2,h4's ncaps coms
    ccap_h1_reslist = [helices[0][1]-i for i in range(4)]
    ccap_h1_com = np.array(get_com(pose_prop_cap, ccap_h1_reslist))

    ncap_h2_reslist = [helices[1][0]+i for i in range(4)]
    ncap_h2_com = np.array(get_com(pose_prop_cap, ncap_h2_reslist))

    caps_com_left = (ccap_h1_com + ccap_h3_com + ncap_h2_com + ncap_h4_com)/4                

    # compute tilt_direction_range: angle of caps_com_left - com_h3ccap - caps_com
    vec_h3c_capcom_left = ccap_h3_com - caps_com_left
    vec_h3c_capcom = ccap_h3_com - caps_com
    tilt_direction_range = np.arccos(np.dot(vec_h3c_capcom_left, vec_h3c_capcom) / (np.linalg.norm(vec_h3c_capcom_left) * np.linalg.norm(vec_h3c_capcom)))

    # h3(h3) bottom tilts to center of h6'-h6" (h6-h4), by aligning vectors
    vec_h3n_h3c = ccap_h3_com - ncap_h3_com
    vec_h3n_capcom = caps_com - ncap_h3_com

    new_h3_parent = pose_prop_cap.clone()
    new_h3_parent.delete_residue_range_slow(helices[3][0], new_h3_parent.size())
    new_h3_parent.delete_residue_range_slow(1, helices[1][1])
    #new_h3.dump_pdb('{}{}/tilt_test_h3.pdb'.format(workdir, output_dir))


    tilt_degree_start_num = 1
    if min_tilt_degree < 1 or min_tilt_degree > num_inner_helix_tilt_degree:
        # 0 acceptable (keep the untilt parent scaffold) and below 0 is effectively 0
        if min_tilt_degree != 0 and _DEBUG:
            print('DEBUG:  invalid min_tilt_degree: ', min_tilt_degree, 'acceptable range 0 -', num_inner_helix_tilt_degree, 'will ignore the input value...')
    else:
        tilt_degree_start_num = min_tilt_degree

    # tilt w/ different degrees
    for tilt_degree_ind in range(tilt_degree_start_num, num_inner_helix_tilt_degree+1):


        #tilt_degree_ind = 5

        new_h3 = new_h3_parent.clone()


        # rotate to left
        this_caps_com_left = ccap_h3_com + vec_h3c_capcom_left * (tilt_degree_ind*1.0/num_inner_helix_tilt_degree)
        vec_h3n_this_capcom_left = this_caps_com_left - ncap_h3_com

        ## double check if the source and target vectors are mistakenly switched!
        #R = rotation_matrix_from_vectors(vec_h3n_h3c, vec_h3n_this_capcom_left) 
        R = rotation_matrix_from_vectors(vec_h3n_this_capcom_left, vec_h3n_h3c)
        '''      
        # rotate to right
        this_caps_com = ccap_h3_com + vec_h3c_capcom * (tilt_degree_ind*1.0/num_inner_helix_tilt_degree)
        vec_h3n_this_capcom = this_caps_com - ncap_h3_com

        ## double check if the source and target vectors are mistakenly switched!
        #R = rotation_matrix_from_vectors(vec_h3n_h3c, vec_h3n_this_capcom)
        R = rotation_matrix_from_vectors(vec_h3n_this_capcom, vec_h3n_h3c)
        '''

        rotate_pose(new_h3, numpy_to_rosetta(R))


        #new_h3.dump_pdb('{}{}/tilt_test_h3_rotate.pdb'.format(workdir, output_dir))


        # post rotational translation 
        #   CAUTION: this is weird, as new_h3 should already be aligned at ncap
        #            before rotation, somehow i still need to do this after rotation...
        ncap_new_h3_reslist = list(range(1,5))
        ncap_new_h3_com = np.array(get_com(new_h3, ncap_new_h3_reslist))      
        t = pyrosetta.rosetta.numeric.xyzVector_double_t(*list(ncap_h3_com - ncap_new_h3_com))
        translate_pose(new_h3, t)
        #new_h3.dump_pdb('{}{}/tilt_test_h3_rotate_trans.pdb'.format(workdir, output_dir))

        for tilt_direction_ind in range(num_inner_helix_tilt_direction):

            new_h3_rot = new_h3.clone()
            #new_h3_rot.dump_pdb('{}{}/tilt_rot_left.pdb'.format(workdir, output_dir))

            if tilt_direction_ind > 0:                          

                tilt_direction_angle = tilt_direction_range * tilt_direction_ind * 1.0 / (num_inner_helix_tilt_direction-1)

                # rotate by a hacky way: move new_h3_rot by to origin, rotate along Z-axis, then move back
                t = pyrosetta.rosetta.numeric.xyzVector_double_t(*list(-ncap_h3_com))
                translate_pose(new_h3_rot, t)
                #R = rotation_matrix_along_z_axis(-1*tilt_direction_angle, degree=False)
                R = rotation_matrix_from_axis_vec_and_theta([0,0,1], -1*tilt_direction_angle, radian=True)
                rotate_pose(new_h3_rot, numpy_to_rosetta(R))
                t = pyrosetta.rosetta.numeric.xyzVector_double_t(*list(ncap_h3_com))
                translate_pose(new_h3_rot, t)  

                #new_h3_rot.dump_pdb('{}{}/tilt_rot_right.pdb'.format(workdir, output_dir))      


            # build new repeat unit (h4, new_h3_rot, h6)
            pose_prop_cap_tilt = pyrosetta.rosetta.core.pose.Pose()
            pose_prop_cap_tilt.append_residue_by_jump(pose_prop_cap.residue(helices[3][0]), 1)
            for source_resid in range(helices[3][0]+1, helices[3][1]+1):
                pose_prop_cap_tilt.append_residue_by_bond(pose_prop_cap.residue(source_resid))
            pose_prop_cap_tilt.append_residue_by_jump(new_h3_rot.residue(1), pose_prop_cap_tilt.size())
            for source_resid in range(2, new_h3_rot.size()+1):
                pose_prop_cap_tilt.append_residue_by_bond(new_h3_rot.residue(source_resid))      
            pose_prop_cap_tilt.append_residue_by_jump(pose_prop_cap.residue(helices[5][0]), pose_prop_cap_tilt.size())
            for source_resid in range(helices[5][0]+1, helices[5][1]+1):
                pose_prop_cap_tilt.append_residue_by_bond(pose_prop_cap.residue(source_resid))
            #pose_prop_cap_tilt.dump_pdb('{}{}/tilt_test_new_unit.pdb'.format(workdir, output_dir))

            #print(repeat_len, num_repeats)
            pose_prop = poorman_repeat_propagate(pose_prop_cap_tilt, repeat_len, num_repeat=num_repeat)            
            #pose_prop.dump_pdb('{}{}/tilt_test_new_unit_prop_tdirect_{}_tdegree_{}.pdb'.format(workdir, output_dir, tilt_direction_ind, tilt_degree_ind))

            output_pose_list.append([pose_prop, tilt_direction_ind, tilt_degree_ind])
            
    return output_pose_list




