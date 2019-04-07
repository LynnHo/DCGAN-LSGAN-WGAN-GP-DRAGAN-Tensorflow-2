import imageio
import pylib as py


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--save_path', default='pics/celeba_dragan.gif')
py.arg('--img_dir', default='output/celeba_gan_dragan/samples_training')
py.arg('--max_frames', type=int, default=0)
args = py.args()

py.mkdir(py.directory(args.save_path))


# ==============================================================================
# =                                  make gif                                  =
# ==============================================================================
# modified from https://www.tensorflow.org/alpha/tutorials/generative/dcgan

with imageio.get_writer(args.save_path, mode='I', fps=8) as writer:
    filenames = sorted(py.glob(args.img_dir, '*.jpg'))
    if args.max_frames:
        step = len(filenames) // args.max_frames
    else:
        step = 1
    last = -1
    for i, filename in enumerate(filenames[::step]):
        frame = 2 * (i**0.3)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
