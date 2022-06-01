import tensorflow as tf
import numpy as np

from lib.math import dot, compute_gradients, alt_secant, secant

class SphereTracer:    
    '''General sphere tracing object.
    
    Attributes:
        sphere_trace_n: int -- number of iterations per trace
        sphere_trace_tolerance: float -- zero-level set tolerance value
        sphere_trace_samples: int -- number of test samples along ray
        sphere_trace_alpha: float -- sdf multiplier for tracing
        secant_n: int -- number of iterations for the secant algorithm
        line_iter: int -- number of iterations to do pushback operation
        line_step: float -- hyperparameter for pushback operation
        normalize_grad_adjust: bool -- whether to normalize gradients for final sphere tracing step
        use_metanlrpp_secant: bool -- if False, uses simple secant
    
    Methods:
        sample -- sample a ray between two points to search for zero-crossing; performs secant
        trace -- perform tracing
        correct_points -- perform gradient direction adjustment
        get_min_pts -- extract minimum-SDF points along rays
        alt_trace -- alternative non-abs tracer
    '''
    
    def __init__(self,
                 sphere_trace_n=16,
                 sphere_trace_tolerance=5e-5,
                 secant_n=8,
                 line_iter=1,
                 line_step=.5,
                 sphere_trace_samples=100,
                 sphere_trace_alpha=1.,
                 normalize_grad_adjust=False,
                 use_metanlrpp_secant=False):
        self.sphere_trace_n = sphere_trace_n
        self.sphere_trace_tolerance = sphere_trace_tolerance
        self.secant_n = secant_n
        self.line_iter = line_iter
        self.line_step = line_step
        self.sphere_trace_samples = sphere_trace_samples
        self.sphere_trace_alpha = sphere_trace_alpha
        self.normalize_grad_adjust = normalize_grad_adjust
        self.use_metanlrpp_secant = use_metanlrpp_secant
        
        if use_metanlrpp_secant:
            self.secant = alt_secant
        else:
            self.secant = secant

    def sample(self, ray, min_dist, max_dist):
        '''Densely sample a ray to find the first zero-crossing.
        
        Arguments:
            ray: Callable -- ray function of the form ray(t) = o + t*d
            min_dist: tf.Tensor -- t-value near bound
            max_dist: tf.Tensor -- t-value far bound
        '''
        sdf = self.sdf
        
        min_dist, max_dist = tf.math.minimum(min_dist, max_dist), tf.math.maximum(min_dist, max_dist)
        
        curr_t = min_dist
        is_active = tf.ones_like(min_dist, dtype=tf.bool)
        
        t_offsets = np.linspace(0, 1, self.sphere_trace_samples)**2
        next_t = curr_t + t_offsets[1]
        
        sdf_val = None
        sdf_prev = None
        total_crossed = ~is_active
        
        for i in range(len(t_offsets)):
            search_t = min_dist + t_offsets[i]
            coords = ray(tf.expand_dims(search_t, -1))
            
            if i == 0:
                sdf_val = sdf(coords)
            else:
                sdf_val = tf.tensor_scatter_nd_update(sdf_val, 
                                                      tf.where(is_active), 
                                                      sdf(tf.boolean_mask(coords, is_active)))
            if i == 0:
                sdf_prev = sdf_val
                total_crossed = tf.reshape(sdf_prev < 0, [-1])
                is_active = is_active & ~total_crossed
                continue
            
            sgn_prev = tf.math.sign(sdf_prev)
            sgn_val = tf.math.sign(sdf_val)
            is_crossed = tf.reshape(sgn_prev != sgn_val, [-1])
            mark_first_crossed = is_active & is_crossed
            total_crossed = total_crossed | mark_first_crossed
            
            updates = tf.boolean_mask(search_t, mark_first_crossed)
            indices = tf.where(mark_first_crossed)
            curr_t = tf.tensor_scatter_nd_update(curr_t, indices, updates - t_offsets[i])
            next_t = tf.tensor_scatter_nd_update(next_t, indices, updates)
            is_active = is_active & ~mark_first_crossed
            
            sdf_prev = sdf_val
            
            too_far = search_t > max_dist
            is_active = is_active & ~too_far

        sec_t = self.secant(curr_t[:, tf.newaxis], next_t[:, tf.newaxis], lambda z : sdf(ray(z)), n=self.secant_n)
        
        return sec_t, total_crossed
    
    def trace(self, r_o, r_d, min_dist, max_dist, current_t, direction=1):
        '''Perform one-directional sphere tracing.
        
        Arguments:
            r_o: tf.Tensor -- ray origins
            r_d: tf.Tensor -- ray directions
            min_dist: tf.Tensor -- t-value near bound
            max_dist: tf.Tensor -- t-value far bound
            current_t: tf.Tensor -- starting t-value
        
        Keyword arguments:
            direction: int -- should be 1 for forward pass or -1 for reverse pass
            tolerance: float -- sphere tracing tolerance hyperparameter (default 5e-5)
        '''
        sdf = self.sdf
        
        last_good_t = current_t
        is_ray_active = tf.ones_like(current_t, dtype=tf.bool)
        
        for i in range(self.sphere_trace_n):
            coords = r_o + tf.expand_dims(current_t, -1)*r_d
            
            if i == 0:
                # calculate sdf for all rays
                coords_sdf = sdf(coords)
            else:
                # only compute sdf for active rays
                coords_sdf = tf.tensor_scatter_nd_update(coords_sdf,
                                                         tf.where(is_ray_active),
                                                         sdf(tf.boolean_mask(coords, is_ray_active)))
            
            # entered surface?
            sign_t = tf.math.sign(coords_sdf)[..., 0]
            is_sign_t_good = direction*sign_t >= 0
            is_good_update = is_sign_t_good & is_ray_active
            last_good_t = tf.tensor_scatter_nd_update(last_good_t, 
                                                      tf.where(is_good_update),
                                                      tf.boolean_mask(current_t, is_good_update))

            # within tolerance?
            is_converged = tf.reshape(tf.math.abs(coords_sdf) < self.sphere_trace_tolerance, [-1])
            is_ray_active = is_ray_active & ~is_converged
            
            # step rays
            current_t = tf.tensor_scatter_nd_update(current_t[:, tf.newaxis],
                                                    tf.where(is_ray_active),
                                                    tf.boolean_mask(current_t, is_ray_active)[:, tf.newaxis] + \
                                                    self.sphere_trace_alpha*direction*tf.boolean_mask(coords_sdf, is_ray_active))
            current_t = tf.reshape(current_t, [-1])
            
            # within bounds?
            is_outside_ndc = (current_t < min_dist) | (current_t > max_dist)
            is_ray_active = is_ray_active & ~is_outside_ndc

        return last_good_t
           
    def dbl_trace(self, r_o, r_d, min_dist, max_dist):
        '''Perform bidrectional sphere tracing with sampling.
        
        Arguments:
            r_o: tf.Tensor -- ray origins
            r_d: tf.Tensor -- ray directions
            min_dist: tf.Tensor -- t-value near bound
            max_dist: tf.Tensor -- t-value far bound
        '''
        sdf = self.sdf
        
        # forward pass
        t_0 = self.trace(r_o, r_d, min_dist, max_dist, min_dist, direction=1)
        
        # check for points that need backwards pass
        min_sdf = sdf(r_o + tf.expand_dims(t_0, -1)*r_d)
        needs_2nd_pass = tf.reshape(tf.math.abs(min_sdf) >= self.sphere_trace_tolerance, [-1])
        needs_2nd_pass = needs_2nd_pass & (t_0 >= min_dist) & (t_0 <= max_dist)
        
        if tf.math.reduce_any(needs_2nd_pass):
            # collect backwards pass elements
            r_o_div = tf.boolean_mask(r_o, needs_2nd_pass)
            r_d_div = tf.boolean_mask(r_d, needs_2nd_pass)
            min_dist_div = tf.boolean_mask(min_dist, needs_2nd_pass)
            max_dist_div = tf.boolean_mask(max_dist, needs_2nd_pass)
            t_0_div = tf.boolean_mask(t_0, needs_2nd_pass)
            
            # backwards pass
            t_1 = self.trace(r_o_div, r_d_div, min_dist_div, max_dist_div, max_dist_div, direction=-1)
            
            # cull rays that went too far
            t_diff = t_1 - t_0_div
            is_lost = t_diff < -1.
            t_1 = tf.tensor_scatter_nd_update(t_1, 
                                              tf.where(is_lost),
                                              tf.boolean_mask(t_0_div, is_lost))
            
            # sample rays
            t_div, total_crossed = self.sample(lambda t : r_o_div + t * r_d_div, t_0_div, t_1)
            
            # update t
            argmin_t = tf.tensor_scatter_nd_update(t_0[:, tf.newaxis], 
                                                   tf.where(needs_2nd_pass),
                                                   t_div)
            # compute cords and sdf
            coords = r_o + argmin_t * r_d
            min_sdf = sdf(coords)
        else:
            # no forward pass
            argmin_t = t_0
            min_sdf = min_sdf
            
            # compute coords
            coords = r_o + tf.expand_dims(argmin_t, -1) * r_d
        
        # convergence test
        argmin_t = tf.reshape(argmin_t, [-1])
        conv_mask = tf.logical_and(
            tf.reshape(tf.math.abs(min_sdf) < 5e-3, [-1]),
            tf.logical_and(argmin_t >= min_dist, argmin_t <= max_dist)
        )
        
        return argmin_t, conv_mask
    
    def correct_points(self, coords_t, r_o, r_d, grad_sdf, sdf_output):
        '''Perform a gradient direction adjustment.
        
        Arguments:
            t_n: tf.Tensor -- a signed distance function
            r_o: tf.Tensor -- ray origins
            r_d: tf.Tensor -- ray directions
            grad_sdf: tf.Tensor -- surface normals
            sdf_output: tf.Tensor -- SDF output for coords
        '''
        # compute the denominator and avoid division by zero
        denom = dot(r_d, grad_sdf, keepdims=True)
        abs_denom = abs(denom)
        clamped_denom = tf.where(abs_denom > 1e-2, abs_denom, 1e-2)
        denom_sgn = tf.math.sign(denom)
        denom_sgn = tf.where(denom_sgn == 0., 1., denom_sgn)
        denom = tf.math.sign(denom) * clamped_denom
        
        # compute adjusted coords
        coords = r_o + (coords_t[:, tf.newaxis] - sdf_output/denom)*r_d
        
        return coords
    
    def final_step(self, trace):
        old_grad_sdf, old_sdf_output = compute_gradients(self.sdf, trace['coords'], 
                                                         normalize=self.normalize_grad_adjust)
        coords = self.correct_points(trace['coords_t'], trace['rays_o'], trace['rays_d'],
                                     old_grad_sdf, old_sdf_output)
        return coords
    
    def get_min_pts(self, rays, t_start, t_end, t_max=40):
        '''Extract minimum-SDF points along a ray.
        
        Arguments:
            rays: tf.tensor -- tensor of size (2, batch_size, 3) containing ray origins and directions, respectively
            t_start: tf.Tensor -- t-value near bound
            t_end: tf.Tensor -- t-value far bound
        
        Keyword arguments:
            t_max: int -- number of samples hyperparameter (default 40)
        '''
        sdf = self.sdf
        
        # broadcast
        t_end = t_end[:, tf.newaxis]
        t_start = t_start[:, tf.newaxis]
        r_o, r_d = tf.unstack(rays)
        
        # densely sample
        t_test = tf.linspace(t_end, t_start, t_max, axis=1)
        pts = tf.expand_dims(r_o, axis=1) + t_test*tf.expand_dims(r_d, axis=1)
        
        # get sdf values
        s_t_test = tf.squeeze(sdf(pts))
        
        # get mins
        s_min = tf.math.argmin(s_t_test, axis=-1)
        min_pts = tf.gather_nd(pts, 
                               tf.stack([tf.range(tf.shape(pts)[0], 
                                                  dtype=tf.int64), s_min], axis=-1))
        min_sdf = tf.math.reduce_min(s_t_test, axis=-1)
            
        return min_pts, min_sdf
    
    def alt_trace(self, r_o, r_d, min_dist, max_dist, run_sampler=True):
        '''Alternative non-abs sphere tracer. Performs one round of bidirectional sphere tracing.
        
        Arguments:
            r_o: tf.Tensor -- ray origins
            r_d: tf.Tensor -- ray directions
            min_dist: tf.Tensor -- t-value near bound
            max_dist: tf.Tensor -- t-value far bound
        
        Keyword arguments:
            run_sampler: bool -- if True, runs sampler in fail cases (default True)
        '''
        sdf = self.sdf
        
        # define the initial t-values
        x_dist = min_dist
        x_f_dist = max_dist
        
        # define the parameterized ray function
        def ray(t, mask=None, expand_rays=False):
            if expand_rays:
                rays_o = tf.expand_dims(r_o, 1)
                rays_d = tf.expand_dims(r_d, 1)
            else:
                rays_o, rays_d = r_o, r_d
            
            if mask is None:
                return rays_o + tf.expand_dims(t, -1)*rays_d
            else:
                return tf.boolean_mask(rays_o, mask) \
                    + t*tf.boolean_mask(rays_d, mask)
        
        # initialize points
        x, x_f = ray(x_dist), ray(x_f_dist)
        
        # compute distance of points
        next_x_sdf = self.sphere_trace_alpha*tf.reshape(sdf(x), -1)
        next_x_f_sdf = self.sphere_trace_alpha*tf.reshape(sdf(x_f), -1)
        
        # initialize masks
        conv_mask = tf.zeros_like(x_dist, dtype=tf.bool)
        conv_f_mask = conv_mask
        int_mask = ~conv_mask
        int_f_mask = ~conv_f_mask
        
        for i in range(self.sphere_trace_n):
            # get converged rays
            conv_mask = conv_mask | tf.reshape(int_mask & (next_x_sdf <= self.sphere_trace_tolerance), -1)
            conv_f_mask = conv_f_mask | tf.reshape(int_f_mask & (next_x_f_sdf <= self.sphere_trace_tolerance), -1)
            
            # update which rays to care about
            int_mask = int_mask & ~conv_mask
            int_f_mask = int_f_mask & ~conv_f_mask

            if not tf.math.reduce_any(int_mask) and not tf.math.reduce_any(int_f_mask):
                    break
            
            # update distances
            x_sdf = tf.where(conv_mask, [0.], tf.reshape(next_x_sdf, -1))
            x_f_sdf = tf.where(conv_f_mask, [0.], tf.reshape(next_x_f_sdf, -1))
                
            # update t-values and points
            x_dist = x_dist + self.sphere_trace_alpha*x_sdf
            x_f_dist = x_f_dist - self.sphere_trace_alpha*x_f_sdf
            
            x = ray(x_dist)
            x_f = ray(x_f_dist)
            
            # get new distance and fix points crossed points
            next_x_sdf = tf.where(int_mask, tf.reshape(sdf(x), -1), 0.)
            next_x_f_sdf = tf.where(int_f_mask, tf.reshape(sdf(x_f), -1), 0.)
            
            crossed_sdf = tf.reshape(next_x_sdf < 0, -1)
            crossed_f_sdf = tf.reshape(next_x_f_sdf < 0, -1)
            
            line_count = 0
            
            while (tf.math.reduce_any(crossed_sdf) or tf.math.reduce_any(crossed_f_sdf)) \
            and line_count < self.line_iter:
                line_scl = (1-self.line_step)/(2**line_count)
                
                # move the points
                if tf.math.reduce_any(crossed_sdf):
                    crossed_idx = tf.where(crossed_sdf)
                    scl_masked_sdf = line_scl * tf.gather(x_sdf, crossed_idx)
                    
                    crossed_dist = tf.gather(x_dist, crossed_idx) - scl_masked_sdf
                    crossed_pts = ray(crossed_dist, crossed_sdf)
                    crossed_sdf = tf.reshape(sdf(crossed_pts), -1)
                    
                    x_dist = tf.tensor_scatter_nd_update(x_dist, 
                                                         crossed_idx, 
                                                         tf.reshape(crossed_dist, -1))
                    
                    x = tf.tensor_scatter_nd_update(x, crossed_idx, crossed_pts)
                    next_x_sdf = tf.tensor_scatter_nd_update(next_x_sdf, crossed_idx, crossed_sdf)
                    
                    # get new distances and update mask
                    crossed_sdf = tf.reshape(next_x_sdf < 0, -1)
                
                if tf.math.reduce_any(crossed_f_sdf):
                    crossed_f_idx = tf.where(crossed_f_sdf)
                    scl_masked_f_sdf = line_scl * tf.gather(x_f_sdf, crossed_f_idx)
                    
                    crossed_f_dist = tf.gather(x_f_dist, crossed_f_idx) + scl_masked_f_sdf
                    crossed_f_pts = ray(crossed_f_dist, crossed_f_sdf)
                    crossed_f_sdf = tf.reshape(sdf(crossed_f_pts), -1)
                    
                    x_f_dist = tf.tensor_scatter_nd_update(x_f_dist, 
                                                           crossed_f_idx, 
                                                           tf.reshape(crossed_f_dist, -1))
                    x_f = tf.tensor_scatter_nd_update(x_f, crossed_f_idx, crossed_f_pts)
                    next_x_f_sdf = tf.tensor_scatter_nd_update(next_x_f_sdf, crossed_f_idx, crossed_f_sdf)
                    
                    # get new distances and update mask
                    crossed_f_sdf = tf.reshape(next_x_f_sdf < 0, -1)
            
                line_count += 1
            
            # discard out of bounds points
            int_mask = int_mask & (x_dist <= max_dist) & (x_dist >= min_dist)
            int_f_mask = int_f_mask & (x_f_dist >= min_dist) & (x_f_dist <= max_dist)
    
        if tf.math.reduce_any(int_mask) and run_sampler:
            masked_x_dist = tf.boolean_mask(x_dist, int_mask)
            masked_x_f_dist = tf.boolean_mask(x_f_dist, int_mask)
            
            sec_t, is_crossed = self.sample(lambda z : ray(z, mask=int_mask), masked_x_dist, masked_x_f_dist)
            conv_mask = tf.tensor_scatter_nd_update(conv_mask, 
                                                    tf.where(int_mask),
                                                    is_crossed)
            
            x_dist = tf.tensor_scatter_nd_update(x_dist[:, tf.newaxis], tf.where(int_mask), sec_t)
            x_dist = tf.reshape(x_dist, [-1])
            
            conv_mask = conv_mask & (x_dist >= min_dist) & (x_dist <= max_dist)
        else:
            conv_mask = conv_mask & (x_dist >= min_dist) & (x_dist <= max_dist)

        return x_dist, conv_mask
