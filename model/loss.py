import tensorflow as tf

def generalized_mean_norm(val, gt=None, p=2, axis=-1):
    '''Compute the mean of ||input||^p, where ||.|| denotes Lp norm.
    
    Arguments:
        val: tf.Tensor -- input to be normed and averaged
        
    Keyword arguments:
        gt: tf.Tensor -- ground truth; if not None, subtracted with val
        p: int -- order of norm
        axis: int -- axis to norm over
    '''
    if gt is not None:
        val = val - gt
        
    val = tf.norm(val, ord=p, axis=axis)
    
    return tf.math.reduce_mean(val**p)

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.float32)))
def soft_mask_loss(s_min, masks, alpha=50.):
    '''Compute soft mask loss proposed in Yariv et al. (2020).
    
    Arguments:
        s_min: tf.Tensor -- minimal SDF values along rays
        masks: tf.Tensor -- corresponding foreground/background masks for points
        
    Keyword arguments:
        alpha: float or tf.Tensor -- softness parameter (default 50.)
    '''
    s_sig = tf.math.sigmoid(-alpha * s_min)
    s_bce_sig = tf.keras.metrics.binary_crossentropy(tf.expand_dims(masks, -1), s_sig)

    return tf.math.reduce_mean(s_bce_sig, axis=-1)/alpha