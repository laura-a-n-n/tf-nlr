import tensorflow as tf
import numpy as np

''' Tensor transformations '''

def conjugate(m, p):
    '''Conjugate the first matrix m by the second matrix p.'''
    return p @ m @ tf.linalg.inv(p)

def dot(x, y, keepdims=False):
    '''Reduce the last dimension of a tensor by performing a dot product.'''
    return tf.reduce_sum(x * y, axis=-1, keepdims=keepdims)

def transform_vectors(mat, vec):
    '''Transform a tensor of vectors by a matrix.'''
    return (mat @ vec[..., None])[..., 0]

def squish(x):
    '''Pass through a bijection [-1, 1] -> [0, 1].'''
    return (x / 2) + .5

def fourier_features(x: tf.Tensor, k=4, p_fns=[tf.math.sin, tf.math.cos]):
    '''Compute a stack of 3d tensors from k pairs of the form [sin(2kpi x), cos(2kpi x)].'''
    r = [x]

    for i in range(1, k+1):
        for f in p_fns:
            r.append(tf.reshape(f(2.*i * np.pi * x), [-1, 3]))

    return tf.stack(r, axis=1)

''' Intersection and root tests '''

def get_3d_line_int(o: tf.Tensor, d: tf.Tensor, min_det=1e-5, interpolation=.5):
    '''Attempt least squares line intersection approximation.'''
    a = tf.zeros((3, 3))
    b = tf.zeros((3, 1))

    d = tf.transpose(tf.transpose(d) / tf.linalg.norm(d, axis=1))

    for i, origin_i in enumerate(o):
        uu = d[i][..., tf.newaxis] * d[i][tf.newaxis, ...]
        id_uu = tf.eye(3) - uu
        a += id_uu
        b += id_uu @ tf.expand_dims(origin_i, -1)

    if tf.linalg.det(a) < min_det:
        raise RuntimeError('Could not determine the lines intersection')
    else:
        return tf.linalg.inv(a) @ b
    
def get_sphere_intersection(r_o, r_d, radius=1, close=None):
    '''Compute intersections of rays with a sphere of a given radius.'''
    b = 2 * dot(r_o, r_d, keepdims=True)
    c = tf.linalg.norm(r_o, axis=-1, keepdims=True)**2 - radius**2

    # we allow a negative discriminant to return NaN for no solns
    numerator = 2*c
    discriminant_sqrt = tf.math.sqrt(b**2 - 4*c)
    denom_1 = -b + discriminant_sqrt
    denom_2 = -b - discriminant_sqrt

    soln_1 = numerator/denom_1
    soln_2 = numerator/denom_2

    closer = tf.where(abs(soln_1) < abs(soln_2), soln_1, soln_2)
    further = tf.where(abs(soln_1) > abs(soln_2), soln_1, soln_2)

    if close is None:
        return closer, further
    elif close:
        return closer
    else:
        return further

def secant(x_0: tf.Tensor, x_1: tf.Tensor, f, n=8, clamp_radius=1e-10):
    '''Approximate roots of a function f between two points x_0, x_1 for n iterations.

    Arguments:
        x_0: tf.Tensor -- lower bound of the desired zero
        x_1: tf.Tensor -- upper bound of the desired zero
        f: Callable -- the function whose roots are to be approximated
    
    Keyword arguments:
        n: int -- the number of iterations
        clamp_radius: float -- radius to clamp by if abs(denom) - clamp_radius >= 0 (default 1e-10)
    '''
    for step in range(n):
        denom = f(x_1) - f(x_0)
        denom_sgn = tf.math.sign(denom)
        denom_sgn = tf.where(denom_sgn == 0., 1., denom_sgn)
        denom = denom_sgn * tf.where(abs(denom) - clamp_radius >= 0, 
            abs(denom), clamp_radius)

        x_2 = x_1 - f(x_1) * (x_1 - x_0)/denom
        x_0, x_1 = x_1, x_2

    return x_2

def alt_secant(t_0: tf.Tensor, t_1: tf.Tensor, f, n=8, clamp_radius=1e-5):
    '''Alternative secant of a function f between two points x_0, x_1 for n iterations; converted from MetaNLR++ PyTorch code.

    Arguments:
        x_0: tf.Tensor -- lower bound of the desired zero
        x_1: tf.Tensor -- upper bound of the desired zero
        f: Callable -- the function whose roots are to be approximated
    
    Keyword arguments:
        n: int -- the number of iterations (default 8)
        clamp_radius: float -- radius to clamp by if abs(denom) - clamp_radius >= 0 (default 1e-5)
    '''
    mid_t = t_0
    argmin_t = t_0
    argmin_t_f = tf.fill(t_0.shape, 1e20)
    
    f_0 = f(t_0)
    f_1 = f(t_1)
    
    for i in range(n):
        denom = f_1 - f_0
        mid_t = t_0 - f_0 * (t_1 - t_0) / denom
        
        nan_mask = abs(denom) < clamp_radius
        nan_replace = .5 * (tf.boolean_mask(t_0, nan_mask) + tf.boolean_mask(t_1, nan_mask))
        mid_t = tf.tensor_scatter_nd_update(mid_t, tf.where(nan_mask), nan_replace)
        
        f_mid = f(mid_t)
        
        is_new_min = abs(f_mid) < abs(argmin_t_f)
        argmin_t = tf.tensor_scatter_nd_update(argmin_t, 
                                               tf.where(is_new_min), 
                                               tf.boolean_mask(mid_t, is_new_min))
        argmin_t_f = tf.tensor_scatter_nd_update(argmin_t_f,
                                                   tf.where(is_new_min),
                                                   tf.boolean_mask(f_mid, is_new_min))
        
        is_left = f_0 * f_mid < 0
        t_1 = tf.tensor_scatter_nd_update(t_1, tf.where(is_left),
                                          tf.boolean_mask(mid_t, is_left))
        
        is_right = f_1 * f_mid < 0
        t_0 = tf.tensor_scatter_nd_update(t_0, tf.where(is_right),
                                          tf.boolean_mask(mid_t, is_right))
        
    return argmin_t

''' Random data generation '''

@tf.function
def gen_3d_noise(noise_k=10000, minval=-1., maxval=1.):
    '''Generates 3D noise between minval and maxval in each dimension.
    
    Keyword arguments:
        noise_k: int -- number of points to generate (default 10000)
        minval: float -- minimum component value (default -1.)
        maxval: float -- minimum component value (default 1.)
    '''
    return tf.random.uniform(shape=[noise_k, 3], minval=minval, maxval=maxval)

@tf.function
def sphere_data(noise_k=10000, radius=1., both=False):
    '''Return points on the unit sphere.
    
    Keyword arguments:
        noise_k: int -- number of points to generate (default 10000)
        radius: float -- radius of sphere (default 1.)
        both: bool -- if True, returns a tuple of (points, noise) (default False)
    '''
    noise_3d = gen_3d_noise(noise_k=noise_k)

    # get points on the unit sphere
    sphere = radius * noise_3d/tf.linalg.norm(noise_3d, axis=-1, keepdims=True)

    return both and (sphere, noise_3d) or sphere

''' Gradient computation '''

def compute_gradients(f, coords, normalize=False):
    '''Compute grad f of coords.
    
    Arguments:
        f: Callable -- function to differentiate
        coords: tf.Tensor -- coords to pass through grad f
    
    Keyword arguments:
        normalize: bool -- if True, normalizes gradients if their norm exceeds 1e-5
    '''
    with tf.GradientTape() as g:
        g.watch(coords)
        output = f(coords)
    grads = g.gradient(output, coords)
    
    if normalize:
        grads = normalize_vectors(grads)

    return grads, output

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 3], dtype=tf.float32),))
def normalize_vectors(vecs, nan_thresh=1e-5):
    '''Normalize vectors and avoid NaN.'''
    norms = tf.linalg.norm(vecs, axis=-1)[..., tf.newaxis]
    return tf.where(norms < nan_thresh, vecs, vecs / norms)