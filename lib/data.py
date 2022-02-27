import os
import pathlib
import yaml

import tensorflow as tf

from lib.rays import *
from lib.math import get_sphere_intersection

class Data:
    '''Extract and preprocess data for Neural Lumigraph Rendering.
    
    Usage: Data(path, img_size, rgb='rgb_folder_name', seg='seg_folder_name')
    
    Attributes:
        dataset_path: str -- dataset path (default '.')
        rgb_path: str -- rgb path (default './rgb')
        seg_path: str -- binary mask path (default './mattes')
    
    Methods:
        get_mat -- extracts a list of matrices from YAML format
        compute_rays -- gets rays
        compute_data -- creates a dataset
    '''
    
    def __init__(self, 
                 path,
                 img_size=800,
                 rgb='rgb',
                 seg='mattes',
                 data_type='nlr'):
        '''Initialize data loader.
        
        Arguments:
            path: str -- value for dataset_path attribute
        
        Keyword arguments:
            img_size: int -- longest side length (in px) of resized image (default 800)
            rgb: str -- rgb folder name (default 'rgb')
            seg: str -- binary mask folder name (default 'mattes')
            data_type: str -- type of dataset (default 'nlr')
        '''
        
    def __init__(self, 
                 path,
                 img_size=800,
                 rgb='rgb',
                 seg='mattes',
                 data_type='nlr'):
        '''Initialize data loader.
        
        Arguments:
            path: str -- value for dataset_path attribute
        
        Keyword arguments:
            img_size: int -- longest side length (in px) of resized image (default 800)
            rgb: str -- rgb folder name (default 'rgb')
            seg: str -- binary mask folder name (default 'mattes')
            data_type: str -- type of dataset (default 'nlr')
        '''
        self.dataset_path = path
        self.rgb_path = os.path.join(path, rgb)
        self.seg_path = os.path.join(path, seg)
        self.data_type = data_type
        self.model_matrix = model_matrix
        
        # load data
        rgb_img_names = list(pathlib.Path(self.rgb_path).glob('./*'))
        
        if data_type == 'rho':
            # in case your naming convention is weird
            rgb_img_names = sorted(rgb_img_names, key=lambda i:int(str(i)[str(i).rfind('_')+1:str(i).rfind('.')]))
            seg_img_names = sorted(list(pathlib.Path(self.seg_path).glob('./*')), key=lambda i:int(str(i)[str(i).rfind('_')+1:str(i).rfind('.')]))
        else:
            rgb_img_names = sorted(rgb_img_names)            
            seg_img_names = [pathlib.Path(os.path.join(self.seg_path, name.stem[:-4] + '_mask.png'))
                             for name in rgb_img_names]
        
        # test paths
        if len(rgb_img_names) == 0:
            raise RuntimeError(f'Could not load RGB data from {self.rgb_path}')
        if len(seg_img_names) == 0:
            raise RuntimeError(f'Could not load binary mask data from {self.rgb_path}')
        
        paths_combined = list(zip(rgb_img_names, seg_img_names))
        
        # decode images
        img_rgb = [tf.image.decode_image(tf.io.read_file(str(img_path[0])))
                   for img_path in paths_combined]
        img_seg = [tf.image.decode_image(tf.io.read_file(str(img_path[1])), channels=1)
                   for img_path in paths_combined]
        resolutions = [(img_r.shape[0], img_r.shape[1]) for img_r in img_rgb]

        # resize and clip image tensors
        img_tensors = [
            [
                # rgb
                tf.clip_by_value(
                    tf.image.resize(img[0], [img_size, img_size], 
                    method='bicubic', 
                    antialias=True, 
                    preserve_aspect_ratio=True) / 255, 0., 1.
                ),

                # seg
                (tf.image.resize(img[1], [img_size, img_size], preserve_aspect_ratio=True) / 255) > .5
            ] 
        for img in zip(img_rgb, img_seg)]

        self.rgb_img_names = rgb_img_names
        self.seg_img_names = seg_img_names
        self.img_tensors = img_tensors
        self.resolutions = resolutions
        
        print(f'Images loaded from {path!s}')
        
    def get_mat(self, mat_type, file_list, 
                data_path=None, name='calib_export.yaml', 
                uniform_intrinsics=False, obj_name=None):
        '''Extract a list of matrices from YAML format.'''
        if data_path is None:
            data_path = self.dataset_path

        with open(os.path.join(data_path, name)) as stream:
            try:
                yaml_obj = yaml.safe_load(stream)
                mat_list = []
                
                if mat_type == 'intrinsic' or mat_type == 'extrinsic':
                    obj_name =  mat_type + 's'
                else:
                    obj_name = mat_type
                
                for file_name in file_list:
                    if mat_type == 'intrinsic':
                        if uniform_intrinsics:
                            mat_list.append(
                                tf.constant(
                                    yaml_obj[obj_name]['mat']
                                )
                            )
                        else:
                            mat_list.append(
                                tf.constant(
                                    yaml_obj[file_name.stem]
                                    [obj_name]['camera_matrix']
                                )
                            )
                    else:
                        mat_list.append(
                            tf.constant(
                                yaml_obj[file_name.stem][obj_name]
                            )
                        )
            except yaml.YAMLError as exc:
                print(exc)
        
        return mat_list
    
    def compute_rays(self, 
                     scene_radius_scale=.7,
                     from_cam_pose=False, 
                     uniform_intrinsics=False,
                     view_diag=None, 
                     n=1., 
                     f=1e3):
        '''Set the object's rays attribute to a list of computed rays.'''
        self.rays_near_bound = n
        self.rays_far_bound = f
        
        img_extrinsic = self.get_mat('extrinsic', 
                                     self.rgb_img_names, 
                                     uniform_intrinsics=uniform_intrinsics)
        img_intrinsic = self.get_mat('intrinsic', 
                                     self.rgb_img_names, 
                                     uniform_intrinsics=uniform_intrinsics)
        
        m_views = []
        m_projs = []
        
        for count, img in enumerate(img_extrinsic):
            # get image width and height
            resolution = self.resolutions[count]
            extr = img_extrinsic[count]
            intr = img_intrinsic[count]
            
            if from_cam_pose:
                R = extr[:3, :3]
                extr_3_4 = tf.concat([R, -R @ extr[:3, -1, tf.newaxis]], axis=-1)
                extr = tf.concat([extr_3_4, tf.eye(4)[tf.newaxis, -1, :]], axis=0)
            
            view_mat, proj_mat = params_to_gl(extr, intr, resolution, view_diag=view_diag, n=n, f=f)
            
            assert tf.math.abs(1 - tf.linalg.det(view_mat)) < 1e-2
           
            m_views += [view_mat]
            m_projs += [proj_mat]
            
        self.m_views = m_views
        self.m_projs = m_projs
        
        model_mat = self.model_matrix(m_views, scene_radius_scale=scene_radius_scale)

        rays_o_list = []
        rays_d_list = []
        
        for count, item in enumerate(zip(m_views, m_projs)):
            view_mat, proj_mat = item
            resolution = tf.convert_to_tensor(self.img_tensors[count][0].shape[:2], dtype=tf.int64)
            
            rays_o, rays_d = get_rays(resolution, model_mat, view_mat, proj_mat)
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)

        self.rays = [rays_o_list, rays_d_list]
    
        tf.print('Rays generated')
    
    def compute_dataset(self, v_img=-1) -> tf.data.Dataset:
        '''Set the object's dataset attribute to a tf.data.Dataset.'''
        n_imgs = len(self.img_tensors)
        
        if v_img >= n_imgs:
            raise RuntimeError('Invalid validation image index')
        v_img = (n_imgs+v_img) % n_imgs
        self.v_img = v_img
        
        dataset = []
        self.orig_data = []
            
        for count, img in enumerate(self.img_tensors):
            flattened_rays = []
            for ray_set in self.rays:
                flattened_rays.append(tf.reshape(ray_set[count], [-1, 3]))
            
            # flatten the image tensors
            flattened_rgb = tf.reshape(img[0], [-1, 3])
            flattened_seg = tf.reshape(img[1], [-1, 1])

            # get sphere intersections
            close_int, far_int = get_sphere_intersection(flattened_rays[0], flattened_rays[1])
            
            # insert into dataset
            img_tensor = tf.concat([flattened_rgb, 
                                    *flattened_rays, 
                                    tf.cast(flattened_seg, dtype=tf.float32),
                                    close_int,
                                    far_int], axis=-1)
            self.orig_data.append(img_tensor)
            
            if count != v_img:
                # cull tensors whose rays don't intersect scene bounds             
                cull_mask = tf.squeeze(~tf.math.is_nan(close_int))
                img_tensor = tf.boolean_mask(img_tensor, cull_mask)
                
                dataset.append(img_tensor)
            else:
                self.validation_dataset = tf.data.Dataset.from_tensor_slices(img_tensor)

        self.dataset = tf.data.Dataset.from_tensor_slices(tf.concat(dataset, axis=0))
        
        tf.print('Dataset generated')
        
        return self.dataset