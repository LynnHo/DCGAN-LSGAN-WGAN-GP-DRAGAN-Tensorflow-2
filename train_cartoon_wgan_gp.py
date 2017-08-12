from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
import models_64x64 as models


""" param """
epoch = 100
batch_size = 64
lr = 0.0002
z_dim = 100
n_critic = 5
gpu_id = 1

''' data '''
# you should prepare your own data in ./data/faces
# cartoon faces original size is [96, 96, 3]


def preprocess_fn(img):
    re_size = 64
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/faces/*.jpg')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[96, 96, 3], preprocess_fn=preprocess_fn)


""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' models '''
    generator = models.generator
    discriminator = models.discriminator_wgan_gp

    ''' graph '''
    # inputs
    real = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # generate
    fake = generator(z, reuse=False)

    # dicriminate
    r_logit = discriminator(real, reuse=False)
    f_logit = discriminator(fake)

    # losses
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    gp = gradient_penalty(real, fake, discriminator)
    d_loss = -wd + gp * 10.0
    g_loss = -tf.reduce_mean(f_logit)

    # otpims
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
    g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    # summaries
    d_summary = utils.summary({wd: 'wd', gp: 'gp'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z, training=False)


""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/cartoon_wgan_gp', sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/cartoon_wgan_gp'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    batch_epoch = len(data_pool) // (batch_size * n_critic)
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # train D
        for i in range(n_critic):
            # batch data
            real_ipt = data_pool.batch()
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})

            save_dir = './sample_images_while_training/cartoon_wgan_gp'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 10, 10), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception, e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
