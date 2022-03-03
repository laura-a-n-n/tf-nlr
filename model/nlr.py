import os
import pickle
import time
import datetime
import warnings

import tensorflow as tf
from tf_siren import SIRENModel

from model import loss
from lib.math import dot, gen_3d_noise, sphere_data, squish, \
                     compute_gradients, normalize_vectors, fourier_features
from lib.rays import compute_depth

import matplotlib.pyplot as plt

class NeuralLumigraph:
    def __init__(self, omega=(30, 30), hidden_omega=(30, 30),
                 s_h5=None, e_h5=None, s_final_units=1,
                 e_final_units=3, s_units=256, e_units=256,
                 latent_code_size=256, s_layers=5, e_layers=5, 
                 posenc_k=4, normalize_grad_input=False,
                 permute_inputs=False):
        '''Initialize a NeuralLumigraph object.
        
        Keyword arguments:
            omega: tuple -- (S initial omega, E initial omega) (default (30, 30))
            hidden_omega: tuple -- (S hidden omega, E hidden omega) (default (30, 30))
            s_h5: str or None -- 'path/to/h5/file' for S (default None)
            e_h5: str or None -- 'path/to/h5/file' for E (default None)
            s_units: int -- number of units per hidden layer for S (default 256)
            e_units: int -- number of units per hidden layer for E (default 256)
            s_final_units: int -- number of final units for S, not including latent code (default 1)
            e_final_units: int -- number of final units for E (default 3)
            latent_code_size: int -- how many units to allocate for the latent feature vector output by S (default 256)
            s_layers: int -- number of total layers for S (default 5)
            e_layers: int -- number of total layers for E (default 5)
            posenc_k: int -- k-value for fourier features (default 4)
            normalize_grad_input: bool -- normalize SDF gradients for input to appearance model? (default False)
            permute_inputs: bool -- if True, arrange inputs according to order used in original NLR paper (default False)
            
        Methods:
            compile -- compile model for training
            bind_data -- bind to a lib.data.Data object
            bind_tracer -- bind to a lib.sphere_tracer.SphereTracer object
            save_model -- save model to path
            load_model -- load model from path
            pretrain -- train neural SDF to a sphere of a given radius
            decode_rgb -- decoding for E output; default perform a bijection [-1, 1] -> [0, 1]
            posenc_features -- perform positional encoding on features
            get_image -- map values into converged pixel slots
            render -- render view
            write_img -- render and plot view
        '''
        self.posenc_k = posenc_k
        self.normalize_grad_input = normalize_grad_input
        self.permute_inputs = permute_inputs
        
        omega_s, omega_e = omega
        hidden_omega_s, hidden_omega_e = hidden_omega

        # create models
        self.sdf = SIRENModel(units=s_units, final_units=s_final_units+latent_code_size, num_layers=s_layers-1, 
                              w0_initial=omega_s, w0=hidden_omega_s, final_activation='linear')
        self.e = SIRENModel(units=e_units, final_units=e_final_units, num_layers=e_layers-1, 
                            w0_initial=omega_e, w0=hidden_omega_e, final_activation='sine')
        
        # separate the SDF from the global feature vector
        self.s = lambda x : self.sdf(x)[..., :1]
        
        # input sizes
        self.posenc_r_d_size = 6*posenc_k
        
        s_input_size = 3
        e_input_size = 9 + self.posenc_r_d_size + latent_code_size
    
        # call the models
        self.s(tf.zeros([1, s_input_size]))
        self.e(tf.zeros([1, e_input_size]))
    
        # load any h5s
        if s_h5 is not None:
            self.sdf.load_weights(s_h5)
            print(s_h5 + ' loaded')
            
        if e_h5 is not None:
            self.e.load_weights(e_h5)
            print(s_h5 + ' loaded')
          
    def compile(self, s_lr=1e-6, e_lr=1e-4):
        '''Compile the model for training.'''
        self.sdf.compile(optimizer=tf.optimizers.Adam(learning_rate=s_lr))
        self.e.compile(optimizer=tf.optimizers.Adam(learning_rate=e_lr))
    
    def bind_data(self, data):
        '''Bind the model to a lib.data.Data object.'''
        self.data = data
    
    def bind_tracer(self, sphere_tracer):
        '''Bind the model to a lib.sphere_tracer.SphereTracer object.'''
        self.sphere_tracer = sphere_tracer
        self.sphere_tracer.sdf = self.s
    
    def save_model(self, path, optimizer=False, overwrite=True):
        '''Save model and possibly optimizer state.'''
        s_path = os.path.join(path, 'S.h5')
        e_path = os.path.join(path, 'E.h5')
        
        if not os.path.exists(path):
            os.makedirs(path)
        elif not overwrite:
            if os.path.exists(s_path):
                raise Exception(str(s_path) + ' exists (overwrite=False)')
            elif os.path.exists(E_path):
                raise Exception(str(e_path) + ' exists (overwrite=False')
        
        self.sdf.save_weights(s_path)
        self.e.save_weights(e_path)
        
        if optimizer:
            warnings.warn('Optimizer saving is not implemented.')
        
        print('Model saved to '+str(path))
        
        return s_path, e_path
    
    def load_model(self, path, optimizer=False):
        '''Load model and possibly optimizer state.'''
        s_path = os.path.join(path, 'S.h5')
        e_path = os.path.join(path, 'E.h5')
        
        if not os.path.exists(path):
            raise Exception(str(path) + ' does not exist')
        elif not os.path.exists(s_path):
            raise Exception(str(s_path) + ' does not exist')
        elif not os.path.exists(e_path):
            raise Exception(str(e_path) + ' does not exist')
        
        self.sdf.load_weights(s_path)
        self.e.load_weights(e_path)
        
        if optimizer:
            warnings.warn('Optimizer loading is not implemented.')
        
        print('Model loaded from '+str(path))
        
        return s_path, e_path
    
    def pretrain(self, radius=.5, noise_k=25000, steps=2000,
                 normal_loss=True, eikonal_loss=True):
        '''Initialize neural SDF to a sphere of a given radius.'''
        for counter in range(steps):
            # generate 3d noise & unit sphere points and concatenate
            sphere, noise_3d = sphere_data(noise_k, radius=.5, both=True)
            sphere_inputs = tf.concat([sphere, noise_3d, noise_3d*1.5], axis=0)

            # compute the ground truth
            sphere_labels = tf.linalg.norm(sphere_inputs, axis=-1) - radius
            
            # initialize loss values
            loss_err, l_n, l_e = 0., 0., 0.

            with tf.GradientTape() as tape:
                # normal loss
                if normal_loss:
                    grad_s, sdf_output = compute_gradients(self.s, sphere)
                    l_n = .02*loss.generalized_mean_norm(grad_s, gt=sphere/radius)
                
                # ground truth loss
                out = self.s(sphere_inputs)
                loss_err = .6*loss.generalized_mean_norm(out, gt=sphere_labels[:, tf.newaxis], p=1)
                
                # eikonal loss
                if eikonal_loss:
                    grad_s_norm = tf.linalg.norm(grad_s, axis=-1)
                    l_e = .01*tf.keras.metrics.mean_squared_error(1., grad_s_norm)
                
                # sum
                loss_val = loss_err + l_e + l_n

            self.sdf.optimizer.minimize(loss_val, self.sdf.trainable_variables, tape=tape)

            tf.print(f'SDF loss {loss_err:.6f} eikonal loss {l_e:.3f} normal loss {l_n:.3f} total {loss_val:.3f}')
    
    def decode_rgb(self, rgb):
        '''Pass through a bijection [-1, 1] -> [0, 1].'''
        return squish(rgb)
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 3], dtype=tf.float32),))
    def posenc_features(self, features):
        '''Compute Fourier features.'''
        if self.permute_inputs:
            p_fns = [tf.math.cos, tf.math.sin]
        else:
            p_fns = [tf.math.sin, tf.math.cos]
        
        return fourier_features(features, k=self.posenc_k, p_fns=p_fns)
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 3], dtype=tf.float32), 
                                  tf.TensorSpec(shape=[None], dtype=tf.bool),
                                  tf.TensorSpec(shape=[None, 3], dtype=tf.float32),))
    def get_image(self, patch, conv_mask, rgb, masks=None):
        '''Map colors rgb into their correct slots in patch.'''
        if masks is not None:
            conv_mask = tf.tensor_scatter_nd_update(masks, tf.where(masks), conv_mask)
            
        x_indices = tf.where(conv_mask)
        
        return tf.tensor_scatter_nd_add(patch, x_indices, self.decode_rgb(rgb)-1)
    
    def unpack_and_trace(self, 
                         batch, 
                         transform_rays_o=tf.identity, 
                         transform_rays_d=tf.identity, 
                         training=False):
        '''Unpack data and run normal sphere tracing.'''
        # unpack data
        batch_shape = tf.shape(batch)
        
        if training:
            # get only foreground rays
            masks = tf.equal(batch[:, 9], 1.)
            px_batch = tf.boolean_mask(batch, masks)
        else:
            # trace all rays
            px_batch = batch

        all_rays_o = transform_rays_o(px_batch[:, 3:6])
        all_rays_d = transform_rays_d(px_batch[:, 6:9])
        min_dist = px_batch[:, 10]
        max_dist = px_batch[:, 11]

        # sphere trace
        coords_t, conv_mask = self.sphere_tracer.dbl_trace(all_rays_o, all_rays_d,
                                                           min_dist, max_dist)

        # mask
        rays_o = tf.boolean_mask(all_rays_o, conv_mask)
        rays_d = tf.boolean_mask(all_rays_d, conv_mask)

        # get converged surface points
        coords_t = tf.boolean_mask(coords_t, conv_mask)
        coords = rays_o + coords_t[:, tf.newaxis] * rays_d

        trace_dict = {'coords': coords, 
                      'coords_t': coords_t,
                      'conv_mask': conv_mask,
                      'rays_o': rays_o,
                      'rays_d': rays_d}
        
        output_dict = {'batch_shape': batch_shape,
                       'trace': trace_dict}
        
        if training:
            colors = tf.boolean_mask(px_batch[:, :3], conv_mask)
            output_dict['colors'] = colors
            
            # prepare l_m_input
            bg_batch = tf.boolean_mask(batch[:, 3:], ~masks)
            l_m_input = tf.reshape(bg_batch[:, :6], [-1, 2, 3])
            l_m_masks = bg_batch[:, 6]
            l_m_start = bg_batch[:, 7]
            l_m_end = bg_batch[:, 8]
            
            unconv_mask = ~conv_mask
            num_unconv = tf.math.reduce_sum(tf.cast(unconv_mask, tf.int32))
            if num_unconv > 0:
                # get the rays that didn't converge and put them with the input to L_M
                unconv_batch = tf.boolean_mask(tf.stack([all_rays_o, all_rays_d], 1), unconv_mask)
                l_m_start = tf.concat([l_m_start, tf.boolean_mask(min_dist, unconv_mask)], axis=0)
                l_m_end = tf.concat([l_m_end, tf.boolean_mask(max_dist, unconv_mask)], axis=0)
                l_m_input = tf.concat([l_m_input, unconv_batch], axis=0)
                l_m_masks = tf.concat([l_m_masks, tf.ones([num_unconv], dtype=l_m_masks.dtype)], axis=0)
            l_m_input = tf.transpose(l_m_input, [1, 0, 2])

            output_dict['mask_loss'] = {'l_m_input': l_m_input,
                                        'l_m_start': l_m_start,
                                        'l_m_end': l_m_end,
                                        'l_m_masks': l_m_masks}
        else:
            output_dict['trace']['coords'] = self.sphere_tracer.final_step(trace_dict)
        
        return output_dict
    
    def get_appearance_features(self, trace, coords):
        # recompute normals and compute feature vector
        with tf.GradientTape() as g:
            g.watch(coords)
            s_output = self.sdf(coords)
            sdf_output = s_output[:, :1]

        grad_sdf = g.gradient(sdf_output, coords)
        sdf_feature_vector = s_output[:, 1:]

        if self.normalize_grad_input:
            grad_sdf = normalize_vectors(grad_sdf)

        # positional encoding for r_d
        posenc_rays_d = self.posenc_features(trace['rays_d'])
        posenc_rays_d = tf.reshape(posenc_rays_d, [tf.shape(coords)[0], -1])
        
        if self.permute_inputs:
            return coords, posenc_rays_d[:, :3], grad_sdf, sdf_feature_vector, posenc_rays_d[:, 3:]
        else:
            return coords, grad_sdf, posenc_rays_d, sdf_feature_vector
    
    def compute_losses(self, tape, opt, batch, trace, rgb, smoothness_loss_input, mask_loss):
        # appearance losses
        with tf.device(opt.training['device']['l_r_device']):
            l_r = loss.generalized_mean_norm(self.decode_rgb(rgb), gt=batch['colors'], p=1)
        with tf.device(opt.training['device']['l_s_device']):
            l_s = loss.generalized_mean_norm(smoothness_loss_input)

        # sampling
        with tf.device(opt.training['device']['sampling_device']):
            with tape.stop_recording():
                sampling_time = time.time()

                if opt.training['mask_loss']['batch_sampling']:
                    # split into two batches
                    midpoint = (mask_loss['l_m_start'] + mask_loss['l_m_end'])/2
                    s_pts_0, s_min_0 = self.sphere_tracer.get_min_pts(mask_loss['l_m_input'], mask_loss['l_m_start'], midpoint, 
                                                                      t_max=opt.training['mask_loss']['num_samples']//2)
                    s_pts_1, s_min_1 = self.sphere_tracer.get_min_pts(mask_loss['l_m_input'], midpoint, mask_loss['l_m_end'], 
                                                                      t_max=opt.training['mask_loss']['num_samples']//2)
                    s_pts = tf.where((s_min_0 < s_min_1)[:, tf.newaxis], s_pts_0, s_pts_1)
                else:
                    # one batch
                    s_pts, s_min = self.sphere_tracer.get_min_pts(**mask_loss, t_max=opt.training['mask_loss']['num_samples'])

                if opt.training['print_times']:
                    tf.print('Sampling took {:.5f}s'.format(time.time() - sampling_time))

        # soft mask loss
        with tf.device(opt.training['device']['l_m_device']):
            l_m_time = time.time()

            s_min = self.s(s_pts)
            l_m = loss.soft_mask_loss(s_min, mask_loss['l_m_masks'], alpha=mask_loss['alpha'])

            if opt.training['print_times']:
                tf.print('Soft mask loss took {:.5f}s'.format(time.time() - l_m_time))

        # eikonal loss
        with tf.device(opt.training['device']['l_e_device']):
            eikonal_time = time.time()

            with tape.stop_recording():
                l_e_input = gen_3d_noise(noise_k=batch['batch_shape'][0])

            l_e_grads, _ = compute_gradients(self.s, l_e_input)
            l_e_norms = tf.linalg.norm(l_e_grads, axis=-1)
            l_e = tf.keras.metrics.mean_squared_error(1., l_e_norms)

            if opt.training['print_times']:
                tf.print('Eikonal took {:.5f}s'.format(time.time() - eikonal_time))

        # compute final loss
        l_e *= opt.training['loss_weights']['w_e']
        l_m *= opt.training['loss_weights']['w_m']
        l_s *= opt.training['loss_weights']['w_s']
        loss_val = l_r + l_e + l_m + l_s
        loss_dict = {'l_r': l_r, 'l_m': l_m, 'l_e': l_e, 'l_s': l_s}
        
        return loss_val, loss_dict
    
    def render(self, rendering_opt=None, v_img=None, batch_size=50000, 
               compute_depth_img=False, compute_normal_img=False, 
               transform_rays_o=tf.identity, transform_rays_d=tf.identity, 
               verbose=True):
        '''Render view by view index.'''
        rendered_img = []
        depth_img = []
        normal_img = []
        
        view_mat = self.data.m_views[self.data.v_img]
        proj_mat = self.data.m_projs[self.data.v_img]
        n = self.data.rays_near_bound
        f = self.data.rays_far_bound
        
        if compute_normal_img:
            light_dir = tf.constant(rendering_opt['light_dir'])
            normal_bias = rendering_opt['normal_bias']
        
        if v_img is None:
            v_img = self.data.v_img
        
        view_dataset = tf.data.Dataset.from_tensor_slices(self.data.orig_data[v_img]).batch(batch_size)
        
        if verbose:
            tf.print('Rendering view')
            num_steps = len(view_dataset)
        
        for step, px_batch in enumerate(view_dataset):
            if verbose:
                tf.print('Render step {} of {}'.format(step+1, num_steps))
            
            # unpack data
            batch = self.unpack_and_trace(px_batch, 
                                          transform_rays_o,
                                          transform_rays_d,
                                          training=False)
            trace = batch['trace']
            coords = trace['coords']
            
            # get appearance features
            e_input = self.get_appearance_features(trace, coords)
            
            # get output
            rgb = self.e(tf.concat([*e_input], axis=-1))

            # map colors into their slots
            patch = tf.ones([batch['batch_shape'][0], 3])
            patch = self.get_image(patch, trace['conv_mask'], rgb)
            
            # add to image
            rendered_img.append(patch)

            if compute_depth_img:
                # compute depth map
                x = tf.tensor_scatter_nd_update(tf.zeros_like(patch), tf.where(trace['conv_mask']), coords)

                depth = compute_depth(x, view_mat, proj_mat, n, f)
                conv_depth = tf.boolean_mask(depth, trace['conv_mask'])
                max_depth = tf.math.reduce_max(conv_depth)
                depth = tf.clip_by_value(depth, 0., max_depth)

                depth = tf.where(trace['conv_mask'], depth, 0.)
                depth_img.append(depth)

            if compute_normal_img:
                # compute normal map
                grad_sdf = e_input[1+self.permute_inputs]
                normal_map = tf.zeros([tf.shape(trace['conv_mask'])[0], 3], dtype=grad_sdf.dtype)
                if tf.shape(grad_sdf)[0] > 0:
                    normal_map = tf.tensor_scatter_nd_update(normal_map,
                                                             tf.where(trace['conv_mask']),
                                                             grad_sdf)
                normal_img.append(normal_map)

        final_img = tf.concat(rendered_img, axis=0)
        
        output = [final_img]
        if compute_depth_img:
            depth_img = tf.concat(depth_img, axis=0)
            output.append(depth_img)
        if compute_normal_img:
            normals = tf.concat(normal_img, axis=0)
            normals = normalize_vectors(normals)
            
            if light_dir is not None:
                normal_img = self.decode_rgb(dot(normals, light_dir[tf.newaxis, :]))
            else:
                normal_img = self.decode_rgb(normals)

            # do processing
            normal_img = tf.clip_by_value(255*normal_img + normal_bias, 0, 255)
            normal_img = tf.cast(normal_img, dtype=tf.uint8)
            
            output.append(normal_img)
        if len(output) == 1:
            output = final_img
        else:
            output = tuple(output)
        
        return output
    
    def write_img(self, rendering_opt=None, compute_depth_img=True, compute_normal_img=True,
                  verbose=True, batch_size=50000, v_img=None, write_to_file=True,
                  out_path=None):
        if v_img is None:
            v_img = self.data.v_img
        v_img_data = self.data.img_tensors[v_img][0]
        
        if rendering_opt is None:
            rendering_opt = {
                            # lighting direction for normal image; if None, return RGB normal map
                            'light_dir': [.3202674, -0.91123605, -0.25899315],
                             # brightness parameter for normal image
                            'normal_bias': 70
                            }
        
        # output = [rgb, depth, normal]
        output = self.render(rendering_opt, v_img=v_img, batch_size=batch_size,
                             compute_depth_img=compute_depth_img, 
                             compute_normal_img=compute_normal_img,
                             verbose=verbose)

        f = plt.figure(figsize=(11,4))
        grid_cols = 10*(compute_depth_img+compute_normal_img)

        plt.subplot(121 + grid_cols)
        plt.imshow(tf.reshape(output[0], tf.shape(v_img_data)))
        plt.title('Rendered image')

        plt.subplot(122 + grid_cols)
        plt.imshow(v_img_data)
        plt.title('Source image')

        if compute_depth_img:
            plt.subplot(123 + grid_cols)
            plt.imshow(tf.reshape(output[1], tf.shape(v_img_data)[:2]), cmap='binary')
            plt.title('Rendered depth map')

        if compute_normal_img:
            plt.subplot(123 + grid_cols + compute_depth_img)
            plt.imshow(tf.reshape(output[1+compute_depth_img], tf.shape(v_img_data)[:2]), cmap='binary')
            plt.title('With normal map')

        if write_to_file:
            plt.savefig(out_path)
            f.clear()
            plt.close(f)
            
            return out_path, output
        else:
            return plt, output
