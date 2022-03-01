import os
import argparse
import importlib
import time
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from conf import config

from model import loss
from model.nlr import NeuralLumigraph

from lib.sphere_tracer import SphereTracer
from lib.data import Data
from lib.math import dot, sphere_data, gen_3d_noise, compute_gradients, normalize_vectors

from IPython.display import clear_output

def get_args():
    parser = argparse.ArgumentParser(description='Train a TF-NLR model.')
    
    parser.add_argument('--dataset_path', type=str, default='./data/nlr_dataset/M2', help='Path to dataset root folder.')
    parser.add_argument('--img_ratio', type=int, default=5, help='Scale down images by dividing by this number.')
    parser.add_argument('--checkpoint', type=str, default='h5/pretrain', help='Initialization checkpoint folder.')
    parser.add_argument('--epoch', type=int, default=1, help='Initial epoch count.')
    parser.add_argument('--s_lr', type=float, default=0., help='Initial learning rate for neural SDF. If zero, read from config file.')
    parser.add_argument('--e_lr', type=float, default=0., help='Initial learning rate for appearance model. If zero, read from config file.')
    parser.add_argument('--mask_loss_alpha', type=float, default=None, help='Initial mask loss alpha (type: float or None).')
    parser.add_argument('--gpu', type=str, default='/gpu:0', help='Default GPU device name.')
    parser.add_argument('--cpu', type=str, default='/cpu:0', help='Default CPU device name.')
    parser.add_argument('--out_folder', type=str, default='.', help='Output folder.')

    opt = parser.parse_args()
    opt = argparse.Namespace(**vars(opt), **config.opt)
    
    # device
    for key, value in opt.training['device'].items():
        if value == 'default_gpu':
            opt.training['device'][key] = opt.gpu if tf.test.is_gpu_available() else opt.cpu
        elif value == 'default_cpu':
            opt.training['device'][key] = opt.cpu
    
    # mask loss alpha
    if opt.mask_loss_alpha is not None:
        arg_type = type(opt.mask_loss_alpha)
        
        if arg_type != float:
            raise Exception('Type of argument `mask_loss_alpha` should be float or None, got {}'.format(str(arg_type)))
        else:
            opt.training['mask_loss']['alpha'] = opt.mask_loss_alpha
    
    if opt.s_lr != 0.:
        opt.training['learning_rate']['s_lr']['initial_learning_rate'] = opt.s_lr
    if opt.e_lr != 0.:
        opt.training['learning_rate']['e_lr']['initial_learning_rate'] = opt.e_lr
    
    # paths
    opt.training['validation']['validation_out_path'] = os.path.join(opt.out_folder, opt.training['validation']['validation_out_path'])
    opt.training['checkpoints']['checkpoint_path'] = os.path.join(opt.out_folder, opt.training['checkpoints']['checkpoint_path'])
    opt.training['tensorboard']['log_path'] = os.path.join(opt.out_folder, opt.training['tensorboard']['log_path'])
    
    # make output folder if needed
    if not os.path.exists(opt.out_folder):
            os.makedirs(opt.out_folder)
    
    return opt

def train(opt, nlr=None, epoch=1, notebook=False):
    '''Train a TF-NLR model.'''
    if nlr is None:
        ''' Data '''
        
        nlr_data = Data(opt.dataset_path, img_ratio=opt.img_ratio)
        nlr_data.compute_rays(scene_radius_scale=opt.data['scene_radius_scale'])
        nlr_dataset = nlr_data.compute_dataset(v_img=opt.data['v_img'])

        ''' Objects '''
        
        nlr = NeuralLumigraph(**opt.nlr)
        nlr.compile(s_lr=tf.keras.optimizers.schedules.ExponentialDecay(**opt.training['learning_rate']['s_lr']), 
                    e_lr=tf.keras.optimizers.schedules.ExponentialDecay(**opt.training['learning_rate']['e_lr']))
        nlr.load_model(opt.checkpoint)

        sphere_tracer = SphereTracer(**opt.sphere_tracer)
        nlr.bind_tracer(sphere_tracer)
        nlr.bind_data(nlr_data)

    ''' TensorBoard '''
    
    summary_writers = {}

    if opt.training['tensorboard']['write_summaries']:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        for loss_name in opt.training['tensorboard']['losses']:
            log_dir = os.path.join(opt.training['tensorboard']['log_path'], current_time, loss_name)
            summary_writers[loss_name] = tf.summary.create_file_writer(log_dir)

    ''' Loop '''
    
    # loop variables
    alpha = opt.training['mask_loss']['alpha']
    
    # checkpoints
    checkpoint_on_validation = opt.training['checkpoints']['checkpoint_steps'] == 0
    save_checkpoint = False
    
    # validation
    compute_depth_img = opt.training['validation']['compute_depth_img']
    compute_normal_img = opt.training['validation']['compute_normal_img']
    render_verbose = opt.training['validation']['verbose']
    
    # dataset shuffling
    if opt.training['shuffle'] == 0:
        # get dataset size
        opt.training['shuffle'] = tf.data.experimental.cardinality(nlr.data.dataset).numpy()

    while epoch <= opt.training['epochs']:
        tf.print('Epoch {} of {}'.format(epoch, opt.training['epochs']))

        nlr_dataset = nlr.data.dataset.shuffle(opt.training['shuffle']).batch(opt.training['batch_size'])
        tf.print('Dataset shuffled with buffer size {}'.format(opt.training['shuffle']))

        for step, px_batch in enumerate(nlr_dataset):
            batch_time = time.time()

            loss_val, loss_dict, tape = train_step(nlr, opt, px_batch, alpha)

            # check for NaN
            if tf.math.is_nan(loss_val) and opt.training['nan_exception']:
                raise Exception('NaN in loss')

            # optimizer step
            gradient_time = time.time()

            with tf.device(opt.training['device']['optim_device']):
                nlr.sdf.optimizer.minimize(loss_val, nlr.sdf.trainable_variables, tape=tape)
                nlr.e.optimizer.minimize(loss_val, nlr.e.trainable_variables, tape=tape)

            if opt.training['print_times']:
                tf.print('Optimizer step took {:.5f}s'.format(time.time() - gradient_time))

            if opt.training['print_losses']:
                tf.print('loss {:.6f} step {}/{}'.format(loss_val, step+1, len(nlr_dataset)))
                
                for loss_name in loss_dict:
                    tf.print('{} {:.6f}'.format(loss_name, loss_dict[loss_name]))

            if opt.training['print_times']:
                tf.print('Total batch time {:.5f}s'.format(time.time() - batch_time))

            if not checkpoint_on_validation:
                save_checkpoint = ((step+1) % opt.training['checkpoints']['checkpoint_steps']) == 0
                
            # validation
            if step == opt.training['validation']['validation_step'] \
                and epoch % opt.training['validation']['validation_epochs'] == 0 \
                and opt.training['validation']['validate']:
                save_checkpoint = checkpoint_on_validation
                write_to_file = not opt.training['validation']['ipy_show']

                write_img_output, _ = nlr.write_img(opt.rendering, 
                                                    batch_size=opt.training['batch_size'],
                                                    compute_depth_img=compute_depth_img,
                                                    compute_normal_img=compute_normal_img,
                                                    verbose=render_verbose,
                                                    write_to_file=write_to_file,
                                                    out_path=opt.training['validation']['validation_out_path'])
                
                if not write_to_file:
                    # plt.show()
                    write_img_output.show()
            
            # checkpoint
            if opt.training['checkpoints']['write_checkpoints'] and save_checkpoint:
                # save
                checkpoint_name = 'checkpoint_{}_{}'.format(epoch, step+1)
                nlr.save_model(os.path.join(opt.training['checkpoints']['checkpoint_path'], checkpoint_name))

                # disable checkpointing for the next step
                save_checkpoint = False
            
            # clear memory?
            tape = None
            write_img_output = None
                
        # increase mask loss alpha?
        if epoch % opt.training['mask_loss']['alpha_increase'] == 0:
            alpha *= opt.training['mask_loss']['alpha_ratio']

        # TensorBoard logging
        if opt.training['tensorboard']['write_summaries']:
            for writer in summary_writers:
                with summary_writers[writer].as_default():
                    tf.summary.scalar('loss', loss_dict[writer], step=epoch)
        
        # clear output
        if opt.training['print_clear']:
            if notebook:
                clear_output(wait=True)
            else:
                if os.name == 'nt':
                    _ = os.system('cls')
                else:
                    _ = os.system('clear')

        epoch += 1

def train_step(nlr, opt, px_batch, alpha):
    '''Perform one train step.'''
    # unpack data
    batch = nlr.unpack_and_trace(px_batch, training=True)
    trace = batch['trace']
    mask_loss = batch['mask_loss']
    mask_loss['alpha'] = alpha

    # prepare loss
    l_r, l_e, l_m, l_s = 0., 0., 0., 0.
    loss_val = 0.

    with tf.GradientTape(persistent=True) as tape:
        # differentiable sphere tracing
        with tf.device(opt.training['device']['diff_sphere_tracing_device']):
            # compute normals and perform gradient direction adjustment
            coords = nlr.sphere_tracer.final_step(trace)

        # forward
        with tf.device(opt.training['device']['get_features_device']):
            coords, grad_sdf, posenc_rays_d, sdf_feature_vector = nlr.get_appearance_features(trace, coords)
        with tf.device(opt.training['device']['appearance_forward_device']):
            # compute E output
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(posenc_rays_d)

                with tf.GradientTape() as g:
                    g.watch(posenc_rays_d)

                    e_input = tf.concat([coords, grad_sdf, 
                                         posenc_rays_d, sdf_feature_vector], axis=-1)
                    rgb = nlr.e(e_input)

                g_rgb = g.gradient(rgb, posenc_rays_d)
            gg_rgb = gg.gradient(g_rgb, posenc_rays_d)

        # compute loss
        loss_val, loss_dict = nlr.compute_losses(tape, opt, batch, trace, rgb, gg_rgb, mask_loss)
    
    return loss_val, loss_dict, tape

if __name__ == '__main__':
    opt = get_args()
    train(opt, epoch=opt.epoch)
