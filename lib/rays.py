import tensorflow as tf
import numpy as np

from lib.math import dot, conjugate, transform_vectors, get_3d_line_int

def params_to_gl(extrinsic_mat, intrinsic_mat, resolution: tuple, n=1, f=1e3, view_diag=None):
    '''Return view and projection matrices in GL format.'''
    h, w = resolution

    f_x, f_y = tf.linalg.diag_part(intrinsic_mat)[:2]
    c_x, c_y = intrinsic_mat[:2, -1]

    proj = tf.stack([
        [2.*f_x/w, 0., 1. - 2.*c_x/w, 0.],
        [0., 2.*f_y/h, -(1. - 2.*c_y/h), 0.],
        [0., 0., (n+f)/(n-f), 2.*n*f/(n-f)],
        [0., 0., -1., 0.]
    ])

    if view_diag is None:
        view_diag = tf.constant([1., -1., -1., 1.])

    view_transform = tf.eye(4, dtype=tf.float32)
    view_transform = tf.linalg.set_diag(view_transform, view_diag)
    view = conjugate(extrinsic_mat, view_transform)

    return view, proj

def model_matrix(view_matrices, target_radius=.99, scene_radius_scale=.7):
    '''Compute a model matrix from a list of view matrices.'''
    c2ws = [tf.linalg.inv(v) for v in view_matrices]
    cam_pos = [c2w[:3, 3] for c2w in c2ws]
    cam_v_d = [-c2w[:3, 2] for c2w in c2ws]
    cam_int = get_3d_line_int(cam_pos, cam_v_d)
    cam_rad = tf.math.reduce_min(tf.linalg.norm(cam_pos - tf.reshape(cam_int, -1), axis=1))

    radius = cam_rad * scene_radius_scale
    center = cam_int

    translation = -center
    scale = target_radius / radius

    norm_matrix = scale * tf.pad(tf.eye(3, dtype=tf.float32), [[0, 1], [0, 1]])
    norm_matrix += tf.pad(tf.concat([scale*translation, [[1]]], axis=0), [[0, 0], [3, 0]])

    return tf.linalg.inv(norm_matrix)

def get_relative_pixel_locs(resolution: tuple):
    '''Compute relative pixel locations in a given resolution.'''
    h, w = resolution

    w_offset, h_offset = 1./float(w), 1./float(h)

    x, y = tf.meshgrid(tf.linspace(-1 + w_offset, 1 - w_offset, w),
        tf.linspace(1 - h_offset, -1 + h_offset, h))

    return tf.stack([tf.reshape(x, -1), tf.reshape(y, -1)], axis=-1)

def get_rays(resolution: tuple, model_mat, view_mat, proj_mat):
    '''Compute rays from model, view, and projection matrices.'''
    pixels_ndc = get_relative_pixel_locs(resolution)
    pixels_ndc = tf.concat([
            pixels_ndc,
            tf.zeros_like(pixels_ndc[..., :1]),
            tf.ones_like(pixels_ndc[..., :1])
        ], axis=-1)

    cam_space_coords = transform_vectors(tf.linalg.inv(proj_mat), pixels_ndc)
    rays_d_cam = tf.linalg.normalize(cam_space_coords[..., :3], axis=-1)[0]

    model_view_mat = view_mat @ model_mat

    mv_mat_inv = tf.linalg.inv(model_view_mat)

    rays_d = transform_vectors(mv_mat_inv[..., :3, :3], rays_d_cam)
    rays_d = tf.linalg.normalize(rays_d, axis=-1)[0]

    cam_pos = mv_mat_inv[..., :3, 3]
    rays_o = tf.broadcast_to(cam_pos, rays_d.shape)

    return rays_o, rays_d

def compute_depth(coords, view_mat, proj_mat, n=1., f=1e3) -> tf.Tensor:
    '''Compute linear depth for 3D points.'''
    transformation = proj_mat @ view_mat
    ones = tf.ones([*coords.shape[:-1], 1], dtype=coords.dtype)
    coords_w = tf.concat([coords, ones], axis=-1)
    coords_ndc = coords_w @ tf.transpose(transformation)
    
    gl_depth = coords_ndc[..., 2] / coords_ndc[..., 3]
    clip_space_depth = 2.*gl_depth - 1.
    
    linear_depth = 2.*n*f/(f+n-clip_space_depth*(f-n))
    
    return linear_depth