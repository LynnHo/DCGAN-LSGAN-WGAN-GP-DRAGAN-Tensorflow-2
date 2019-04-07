import functools

import imlib as im
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
py.arg('--dataset', default='fashion_mnist', choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
py.arg('--batch_size', type=int, default=64)
py.arg('--epochs', type=int, default=25)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--z_dim', type=int, default=128)
py.arg('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--experiment_name', default='none')
args = py.args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s' % args.gradient_penalty_mode
output_dir = py.join('output', args.experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# setup dataset
if args.dataset in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
    dataset, shape, len_dataset = data.make_32x32_dataset(args.dataset, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = 3

elif args.dataset == 'celeba':  # 64x64
    img_paths = py.glob('data/img_align_celeba', '*.jpg')
    dataset, shape, len_dataset = data.make_celeba_dataset(img_paths, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'anime':  # 64x64
    img_paths = py.glob('data/faces', '*.jpg')
    dataset, shape, len_dataset = data.make_anime_dataset(img_paths, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'custom':
    # ======================================
    # =               custom               =
    # ======================================
    img_paths = ...  # image paths of custom dataset
    dataset, shape, len_dataset = data.make_custom_dataset(img_paths, args.batch_size)
    n_G_upsamplings = n_D_downsamplings = ...  # 3 for 32x32 and 4 for 64x64
    # ======================================
    # =               custom               =
    # ======================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
if args.gradient_penalty_mode in ['dragan', 'wgan-gp']:  # cannot use batch normalization with gradient penalty
    # TODO(Lynn)
    # Layer normalization is more stable than instance normalization here,
    # but instance normalization works in other implementations.
    # Please tell me if you find out the cause.
    d_norm = 'layer_norm'

# networks
G = module.ConvGenerator(input_shape=(1, 1, args.z_dim), output_channels=shape[-1], n_upsamplings=n_G_upsamplings, name='G_%s' % args.dataset)
D = module.ConvDiscriminator(input_shape=shape, n_downsamplings=n_D_downsamplings, norm=d_norm, name='D_%s' % args.dataset)
G.summary()
D.summary()

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)

G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G():
    with tf.GradientTape() as t:
        z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
        x_fake = G(z, training=True)
        x_fake_d_logit = D(x_fake, training=True)
        G_loss = g_loss_fn(x_fake_d_logit)

    G_grad = t.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

    return {'g_loss': G_loss}


@tf.function
def train_D(x_real):
    with tf.GradientTape() as t:
        z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
        x_fake = G(z, training=True)

        x_real_d_logit = D(x_real, training=True)
        x_fake_d_logit = D(x_fake, training=True)

        x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
        gp = gan.gradient_penalty(functools.partial(D, training=True), x_real, x_fake, mode=args.gradient_penalty_mode)

        D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


@tf.function
def sample(z):
    return G(z, training=False)


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G=G,
                                D=D,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
z = tf.random.normal((100, 1, 1, args.z_dim))  # a fixed noise for sampling
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for x_real in tqdm.tqdm(dataset, desc='Inner Epoch Loop', total=len_dataset):
            D_loss_dict = train_D(x_real)
            tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')

            if D_optimizer.iterations.numpy() % args.n_d == 0:
                G_loss_dict = train_G()
                tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                x_fake = sample(z)
                img = im.immerge(x_fake, n_rows=10)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)
