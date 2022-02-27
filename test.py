import os
import argparse

import tensorflow as tf

from model.nlr import NeuralLumigraph
from lib.sphere_tracer import SphereTracer
from lib.data import Data

from conf import config

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a TF-NLR model.')
    
    parser.add_argument('--dataset_path', type=str, default='./data/nlr_dataset/M2', help='Path to dataset root folder.')
    parser.add_argument('--model', type=str, default='h5/M2', help='Model folder.')
    parser.add_argument('--img_size', type=int, default=800, help='Maximum image dimension.')
    parser.add_argument('--out_folder', type=str, default='test', help='Output folder.')
    parser.add_argument('--held_out_only', dest='render_fitted', action='store_false')
    parser.add_argument('--fitted_only', dest='render_held_out', action='store_false')
    parser.add_argument('--no_metrics', dest='metrics', action='store_false')
    parser.set_defaults(render_fitted=True, render_held_out=True, metrics=True)

    opt = parser.parse_args()
    opt = argparse.Namespace(**vars(opt), **config.opt)
    
    # make output folder if needed
    if not os.path.exists(opt.out_folder):
        os.makedirs(opt.out_folder)
    
    return opt

def test(opt, nlr=None):
    tf.print(f'Rendering all views into path {opt.out_folder}')
    
    if nlr is None:
        tf.print('Creating model.nlr.NeuralLumigraph object from config as one was not provided (nlr=None)')
        
        ''' Data '''
        
        nlr_data = Data(opt.dataset_path, img_size=opt.img_size)
        nlr_data.compute_rays(scene_radius_scale=opt.data['scene_radius_scale'])
        nlr_dataset = nlr_data.compute_dataset(v_img=opt.data['v_img'])

        ''' Objects '''
        
        nlr = NeuralLumigraph(**opt.nlr)
        nlr.load_model(opt.model, optimizer=False)

        sphere_tracer = SphereTracer(**opt.sphere_tracer)
        nlr.bind_tracer(sphere_tracer)
        nlr.bind_data(nlr_data)
    
    output = render_all_views(opt, nlr)
    
    if opt.metrics:
        num_rendered, metrics = output
        
        # compute average
        for metric, val in metrics.items():
            avg = val / num_rendered
            
            tf.print(f'Average {metric}: {avg}')
    else:
        num_rendered = output
    
    tf.print('Complete! Rendered {} views for model at {}.'.format(num_rendered, opt.model))

def render_all_views(opt, nlr):
    num_rendered = 0                 
    
    if opt.metrics:
        # NOT IMPLEMENTED: LPIPS; other metrics?
        metrics_dict = {'PSNR': 0., 'SSIM': 0.}
    
    for v_id, view in enumerate(nlr.data.img_tensors):
        should_render = True
        
        if v_id == nlr.data.v_img:
            if not opt.render_held_out:
                should_render = False
                tf.print(f'Skipping view index {v_id} (render_held_out=False)')
            else:
                tf.print(f'Evaluating held out view (index={v_id})')
        elif not opt.render_fitted:
            should_render = False
            tf.print(f'Skipping view index {v_id} (render_fitted=False)')
        else:
            tf.print(f'Evaluating fitted view (index={v_id})')
        
        if should_render:
            _, output = nlr.write_img(opt.rendering, **opt.evaluation['rendering'], 
                                      v_img=v_id, out_path=os.path.join(opt.out_folder, f'{v_id}.png'))
            
            num_rendered += 1
            
            if opt.metrics:
                metrics = compute_metrics(view, output[0])
                
                tf.print(f'Metrics for view {v_id}:')
                for metric, val in metrics.items():
                    metrics_dict[metric] += val
                    
                    tf.print(f'{metric}: {val}')
        
    if opt.metrics:
        return num_rendered, metrics_dict
    else:
        return num_rendered

def compute_metrics(view, output):
    '''Compute metrics.'''
    # constrain to mask
    ground_truth = tf.where(view[1], view[0], tf.ones([1, 3], dtype=view[0].dtype))
    rendered = tf.reshape(output, ground_truth.shape)
    
    psnr = tf.image.psnr(ground_truth, rendered, 1.)
    ssim = tf.image.ssim(ground_truth, rendered, 1.)
    
    return {'PSNR': psnr, 'SSIM': ssim}

if __name__ == '__main__':
    opt = get_args()
    test(opt)