opt = dict(
    nlr = dict(
        omega=(30, 30), # (omega S, omega E) -- see sec. 3.2 of Sitzmann et al. (2020)
        hidden_omega=(30, 30), # (hidden omega S, hidden omega E)
        s_final_units=1, # output units for S
        e_final_units=3, # output units for E
        s_units=256, # units per hidden layer for S
        e_units=256, # units per hidden layer for E
        latent_code_size=256, # size of latent code passed from S to E 
        s_layers=5, # total number of S layers
        e_layers=5, # total number of E layers
        posenc_k=4, # k-value for ray direction Fourier features
        normalize_grad_input=False, # normalize S gradients for input to E?
        permute_inputs=False, # use paper input order?
    ),
    
    data = dict(
        v_img=-1, # validation image index
        scene_radius_scale=.7, # scene radius scale for rays
    ),

    sphere_tracer = dict(
        sphere_trace_n=16, # number of sphere tracing iterations
        sphere_trace_tolerance=5e-5, # zero-level set tolerance value
        sphere_trace_samples=100, # number of test samples along ray
        secant_n=8, # number of iterations for the secant algorithm
        sphere_trace_alpha=1., # sdf multiplier for tracing
        line_iter=1, # number of iterations to do pushback operation (alt_trace only)
        line_step=.5, # hyperparameter for pushback operation (alt_trace only)
        normalize_grad_adjust=False, # normalize gradients for differentiable sphere tracing step?
        use_metanlrpp_secant=False, # use MetaNLR++ secant?
    ),

    training = dict(
        # general
        epochs = 750, # train for this many epochs
        batch_size = 50000, # split the data into batches of this size
        shuffle = 0, # shuffle the data with this buffer size (if 0, automatically set to size of dataset)

        nan_exception = True, # raise exception on loss NaN?
        print_losses = True, # print losses every step?
        print_times = True, # print times?
        print_clear = True, # clear output every epoch?

        # checkpoints
        checkpoints = dict(
            write_checkpoints = True, # save checkpoints?
            checkpoint_steps = 0, # checkpoint every this many steps; if 0, checkpoint on validation
            checkpoint_path = 'h5',
        ),
        
        # TensorBoard
        tensorboard = dict(
            write_summaries=True, # write to TensorBoard?
            losses=['l_r', 'l_m', 'l_e', 'l_s'], # which losses to write
            log_path='logs', # TensorBoard log path
        ),
        
        # learning rate
        learning_rate = dict(
            s_lr=dict(
                initial_learning_rate=1e-6, # S initial learning rate
                decay_steps=30000, # S learning rate decay steps
                decay_rate=.5, # E decay rate
                staircase=True, # E staircase decay
            ),
            
            e_lr=dict(
                initial_learning_rate=1e-4, # E initial learning rate
                decay_steps=30000, # E learning rate decay steps
                decay_rate=.5, # E decay rate
                staircase=True, # E staircase decay
            )
        ),
        
        # mask loss hyperparameters
        mask_loss = dict(
            alpha_increase = 250, # increase alpha every this many epochs
            alpha_ratio = 2., # multiply alpha by this value
            alpha = 50., # initial alpha value
            num_samples = 80, # number of samples along ray to find minimal SDF value
            batch_sampling = True, # if true, does two batches for sampling
        ),

        # loss weights
        loss_weights = dict(
            w_e = 1e-1, # eikonal weight
            w_m = 1e2, # mask loss weight
            w_s = 1e-2, # angular linearization weight
        ),

        # device options
        device = dict(
            diff_sphere_tracing_device = 'default_gpu', # device for the differentiable sphere tracing step
            get_features_device = 'default_gpu', # device for recomputing normals and getting feature vectors
            appearance_forward_device = 'default_gpu', # device for forward-passing to E
            sampling_device = 'default_gpu', # device for ray sampling
            l_r_device = 'default_gpu', # device for color loss
            l_s_device = 'default_gpu', # device for angular smoothness loss
            l_e_device = 'default_cpu', # device for eikonal loss
            l_m_device = 'default_gpu', # device for soft mask loss
            optim_device = 'default_gpu', # device for optimizer step
        ),

        # validation parameters
        validation = dict(
            validate = True, # render validation image?
            validation_step = 0, # which step to validate after?
            validation_epochs = 1, # validate every this many epochs
            compute_depth_img = True, # render depth map?
            compute_normal_img = True, # render image with normal map?
            verbose = True, # verbose validation?
            validation_out_path = 'out.png', # save validation image to this path
            ipy_show = False, # if True, calls matplotlib.pyplot.show instead of saving to file
        ),
    ),
    
    rendering = dict(
        light_dir = [.3202674, -0.91123605, -0.25899315], # lighting direction for normal image; if None, return RGB normal map
        normal_bias = 70, # brightness parameter for normal image
    ),
    
    evaluation = dict(
        rendering = dict(
            batch_size=50000,
            compute_depth_img = True, # render depth map?
            compute_normal_img = True, # render image with normal map?
            verbose = True, # verbose rendering?
        )
    )
)
