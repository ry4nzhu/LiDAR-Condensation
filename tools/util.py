import pickle
import numpy as np
import open3d as o3d

class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram', 'Misc']

def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    assert data is not None
    return data


def save_to_bin(filename, data):
    with open(filename, 'wb') as f:
        print("save to ", filename, data.dtype)
        data.tofile(f)

def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def project_cam2velo(cam, Tr):
    T = np.zeros([4, 4], dtype=np.float32)
    T[:3, :] = Tr
    T[3, 3] = 1
    T_inv = np.linalg.inv(T)
    lidar_loc_ = np.dot(T_inv, cam)
    lidar_loc = lidar_loc_[:3]
    return lidar_loc.reshape(1, 3)

def ry_to_rz(ry):
    angle = -ry - np.pi / 2

    if angle >= np.pi:
        angle -= np.pi
    if angle < -np.pi:
        angle = 2*np.pi + angle

    return angle


class KittiObject(object):
    ''' kitti 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def __str__(self):
        str0 = ('Type, truncation, occlusion, alpha: %s, %d, %d, %f\n' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        str1 = ('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f\n' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        str2 = ('3d bbox h,w,l: %f, %f, %f\n' % \
            (self.h, self.w, self.l))
        str3 = ('3d bbox location, ry: (%f, %f, %f), %f\n' % \
            (self.t[0],self.t[1],self.t[2],self.ry))

        return (str0 + str1 + str2 + str3)


def get_obj_type(obj_str):
    obj_type = -1
    for i in range(len(class_list)):
        if obj_str == class_list[i]:
            obj_type = i
    return obj_type


def read_objs2velo(label_file, Tr_velo2cam):
    '''
    Tr_velo2cam: (3, 4)
    '''

    lines = [line.rstrip() for line in open(label_file)]
    objs_velo = []
    objs_type = []
    for line in lines:
        obj = KittiObject(line)
        if obj.type == 'DontCare':
            continue
        obj_type = get_obj_type(obj.type)
        h = obj.h
        w = obj.w
        l = obj.l
        x = obj.t[0]
        y = obj.t[1]
        z = obj.t[2]
        ry = obj.ry

        rz = ry_to_rz(ry) # ry in camera, rz in velo

        pos_cam = np.ones([4, 1])
        pos_cam[0] = x
        pos_cam[1] = y
        pos_cam[2] = z
        pos_velo = project_cam2velo(pos_cam, Tr_velo2cam) # pos_velo: (1,3)
        x_velo = pos_velo[0][0]
        y_velo = pos_velo[0][1]
        z_velo = pos_velo[0][2]
        obj_velo = [h, w, l, x_velo, y_velo, z_velo, rz, obj_type]
        objs_type.append(obj_type)

        objs_velo.append(obj_velo)
    objs_velo = np.array(objs_velo) #(n, 8)

    return objs_velo, objs_type 


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def get_box_arrow(dim):
    h = dim[0]
    # w = dim[1]
    l = dim[2]
    x = dim[3]
    y = dim[4]
    z = dim[5]
    yaw = dim[6]

    # get direction arrow
    dx = l/2.0*np.cos(yaw)
    dy = l/2.0*np.sin(yaw)
    # a_start = [x, y, z+h]
    # a_end = [x+dx, y+dy, z+h]
    # arrow = [a_start, a_end]
    arrow = [x, y, z+h, x+dx, y+dy, z+h] # [x0, y0, z0, x1, y1, z1], point0--->point1
    return arrow


def box_dim2corners(dim):
    '''
    dim: [h, w, l, x, y, z, yaw]

    8 corners: np.array = n*8*3(x, y, z)
    #         7 -------- 6
    #        /|         /|
    #       4 -------- 5 .
    #       | |        | |
    #       . 3 -------- 2            
    #       |/         |/
    #       0 -------- 1

                ^ x(l)
                |
                |
                |
    y(w)        |
    <-----------O
    '''
    h = dim[0]
    w = dim[1]
    l = dim[2]
    x = dim[3]
    y = dim[4]
    z = dim[5]
    yaw = dim[6]

    # 3d bounding box corners
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    R = rotz(yaw)
    corners_3d = np.dot(R, Box) # corners_3d: (3, 8)

    corners_3d[0,:] = corners_3d[0,:] + x
    corners_3d[1,:] = corners_3d[1,:] + y
    corners_3d[2,:] = corners_3d[2,:] + z

    return np.transpose(corners_3d)


def create_box_from_corners(corners, color=None):
	'''
	corners: 8 corners(x, y, z)
	corners: array = 8*3
	#         7 -------- 6
	#        /|         /|
	#       4 -------- 5 .
	#       | |        | |
	#       . 3 -------- 2
	#       |/         |/
	#       0 -------- 1
	'''
	# 12 lines in a box
	lines = [[0, 1], [1, 2], [2, 3], [3, 0], 
				[4, 5], [5, 6], [6, 7], [7, 4],
				[0, 4], [1, 5], [2, 6], [3, 7]]
	if color == None:
		color = [1, 0, 0] # red
	colors = [color for i in range(len(lines))]
	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector(corners)
	line_set.lines = o3d.utility.Vector2iVector(lines)
	line_set.colors = o3d.utility.Vector3dVector(colors)

	return line_set

def rotation_matrix(roll, yaw, pitch):
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), 
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), 
                   np.cos(pitch)*np.sin(roll), 
                   np.cos(pitch)*np.cos(roll)]])
    return R

def get_open3d_bbox(bbox):
    # KITTI format to open3d
    bbox_new = bbox.copy()
    bbox_new[5] += bbox_new[2] / 2
    # call open3d api
    o3d_bbox = o3d.geometry.OrientedBoundingBox(center=bbox_new[3:6], R=rotation_matrix(0, bbox_new[6], 0), extent=bbox_new[0:3])
    return o3d_bbox


def create_box_from_dim(dim, color=None):
    '''
    dim: list(8) [h, w, l, x, y, z, yaw]
    '''
    box_corners = box_dim2corners(dim)
    box = create_box_from_corners(box_corners, color)
    return box


def seperate_obj_points(points, boxes, box_types):
    #--------------------------------------------------------------
    # create boxes
    #--------------------------------------------------------------
    boxes_o3d = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    all_object_indices = []
    for i in range(boxes.shape[0]):
        dim = boxes[i]
        box_o3d = create_box_from_dim(dim)
        box_o3d = get_open3d_bbox(dim)
        boxes_o3d.append(box_o3d)

        if box_types[i] in [3, 5]: # Pedestrian, Cyclist
            # print(dim, box_types[i])
            
            indices = box_o3d.get_point_indices_within_bounding_box(pcd.points)
            # print(indices)
            all_object_indices += indices
    
    if len(all_object_indices) > 0:
        obj_points = points[np.array(all_object_indices)]
        non_obj_points = np.delete(points, np.array(all_object_indices), axis=0)
    else:
        obj_points = np.zeros([0, 4], dtype=np.float32)
        non_obj_points = points

    return obj_points, non_obj_points