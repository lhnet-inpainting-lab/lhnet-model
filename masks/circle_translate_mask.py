import tensorflow as tf
import numpy as np
import math
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import datetime

from config import Config

FLAGS = Config('./inpaint.yml')
img_shape = FLAGS.img_shapes
IMG_HEIGHT = img_shape[0]
IMG_WIDTH = img_shape[1]

# psnr코드
import numpy

test_num = 0


def psnr(img1, img2):
    input = tf.clip_by_value((img1.numpy() * 0.5 + 0.5), 0., 1.)
    out = tf.clip_by_value((img2.numpy() * 0.5 + 0.5), 0., 1.)
    mse = numpy.mean((input - out) ** 2)
    print("mse: ", mse)
    if mse == 0:
        return 100
    return 10 * math.log10(1. / mse)


# ssim 코드1
from skimage.metrics import structural_similarity as ssim
import imutils
import cv2

# fid 코드1
# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize


# scale an array of images to a new size

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    # print("size of images =", len(images))
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, scale=4.671875, to_shape=new_shape, func='nearest', name='resize')
        # resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False, func='nearest', name='resize')
        # store
        images_list.append(new_image)
        # print("@ ")
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def load(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img)
    return tf.cast(img, tf.float32)


def normalize(img):
    return (img / 127.5) - 1.


def load_image_train(img):
    img = load(img)
    img = resize_pipeline(img, IMG_HEIGHT, IMG_WIDTH)
    return normalize(img)


def resize_pipeline(img, height, width):
    return tf.image.resize(img, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def CSV_reader(input):
    import re
    input = [i.split('tf.Tensor(')[1].split(', shape')[0] for i in input]
    return tf.strings.to_number(input)


def create_mask(FLAGS):
    bbox = random_bbox(FLAGS)
    regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')

    irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
    mask = tf.cast(
        tf.math.logical_or(
            tf.cast(irregular_mask, tf.bool),
            tf.cast(regular_mask, tf.bool),
        ),
        tf.float32
    )
    return mask


def generate_images(input, generator, training=True, url=False, num_epoch=0):
    # input = original
    # batch_incomplete = original+mask
    # stage2 = prediction/inpainted image
    mask = create_mask(FLAGS)
    batch_incomplete = input * (1. - mask)
    stage1, stage2, offset_flow = generator(batch_incomplete, mask, training=training)

    plt.figure(figsize=(30, 30))

    batch_predict = stage2
    batch_complete = batch_predict * mask + batch_incomplete * (1 - mask)

    # input_mask vs stage2_mask

    input_mask = input[0] * mask[0]
    stage2_mask = batch_predict[0] * mask[0]
    '''
    input_mask = input[0] * mask
    stage2_mask = batch_predict[0] * mask
    '''
    # psnr코드2
    # input vs stage2
    cal_psnr = psnr(input[0], batch_predict[0])
    # input vs inpainted image
    cal_psnr1 = psnr(input[0], batch_complete[0])
    # input mask vs stage2 mask
    cal_psnr2 = psnr(input[0] * mask[0], batch_predict * mask[0])
    print('PSNR: input vs stage2 = %.4f' % cal_psnr)
    print('PSNR: input vs inpainted= %.4f' % cal_psnr1)
    print('PSNR: input_mask vs stage2_mask = %.4f' % cal_psnr2)

    # ssim 코드2
    imageA = input[0]
    imageB = batch_complete[0]
    imageC = batch_predict[0]

    imageA = ((imageA.numpy() + 1.) * 127.5).astype("uint8")
    imageB = ((imageB.numpy() + 1.) * 127.5).astype("uint8")
    imageC = ((imageC.numpy() + 1.) * 127.5).astype("uint8")
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    grayC = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(grayA, grayB, full=True)

    print("SSIM: input vs inpainted = {}".format(score))

    (score1, diff1) = ssim(grayA, grayC, full=True)
    print("SSIM: input vs stage2 = {}".format(score1))

    # fid코드2
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # define two fake collections of images
    images1 = input
    images2 = batch_complete
    images3 = batch_predict
    # resize images
    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))
    images3 = scale_images(images3, (299, 299, 3))

    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    images3 = preprocess_input(images3)
    fid = calculate_fid(model, images1, images2)
    fid1 = calculate_fid(model, images1, images3)
    print('FID (input vs inpainted): %.3f' % fid)
    print('FID (input vs stage2): %.3f' % fid1)

    # 교정: 아래 1/2/4/5라인
    display_list = [input[0], batch_incomplete[0], stage1[0], stage2[0], batch_complete[0], offset_flow[0]]
    title = ['Input Image', 'Input With Mask', 'stage1', 'stage2', 'Inpainted Image', 'Offset Flow']
    if not url:
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            title_obj = plt.title(title[i])
            plt.setp(title_obj, color='y')  # set the color of title to red
            plt.axis('off')
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
        if training:
            plt.savefig(f"./images_examples/test_example_{num_epoch}.png")
        else:
            plt.savefig(f"./images_examples/infer_test_example_{num_epoch}__" + datetime.datetime.now().strftime(
                "%H%M%S%f") + ".png")
    else:
        return batch_incomplete[0], batch_complete[0]


def plot_history(g_total_h, g_hinge_h, g_l1_h, d_h, num_epoch, training=True):
    plt.figure(figsize=(20, 10))
    plt.subplot(4, 1, 1)
    plt.plot(g_total_h, label='total_gen_loss')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(g_hinge_h, label='gen_hinge_loss')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(g_l1_h, label='gen_l1_loss')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(d_h, label='dis_loss')
    plt.legend()
    # save plot to file
    if training:
        plt.savefig(f"./images_loss/plot_loss_{num_epoch}.png")
    else:
        plt.savefig(f"./images_loss/infer_plot_loss_{num_epoch}.png")
    plt.clf()
    plt.close()


# COMPUTATIONS
def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., training=True,
                         fuse=True):
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # raw_int_fs[0] = 1
    # raw_int_bs[0] = 1
    # print("raw_int_bs" , raw_int_bs)
    kernel = 2 * rate
    raw_w = tf.image.extract_patches(
        b, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])
    f = resize(f, scale=1. / rate, func='nearest')
    b = resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
               func='nearest')  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1. / rate, func='nearest')
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    # int_fs[0] = 1
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    # int_bs[0] = 1
    w = tf.image.extract_patches(
        b, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.image.extract_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.math.equal(tf.math.reduce_mean(m, axis=[0, 1, 2], keepdims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    scale = softmax_scale
    k = fuse_k
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.math.maximum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1] * bs[2]])

        # softmax to match
        yi *= mm  # mask
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm  # mask

        offset = tf.math.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0),
                                    strides=[1, rate, rate, 1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func='bilinear')
    return y, flow


def random_bbox(FLAGS):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = FLAGS.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - FLAGS.vertical_margin - FLAGS.height
    maxl = img_width - FLAGS.horizontal_margin - FLAGS.width
    t = tf.random.uniform(
        [], minval=FLAGS.vertical_margin, maxval=maxt, dtype=tf.int32)
    l = tf.random.uniform(
        [], minval=FLAGS.horizontal_margin, maxval=maxl, dtype=tf.int32)
    h = tf.constant(FLAGS.height)
    w = tf.constant(FLAGS.width)
    return (t, l, h, w)


def bbox2mask(FLAGS, bbox, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: tuple, (top, left, height, width)

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """

    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h // 2 + 1)
        w = np.random.randint(delta_w // 2 + 1)
        mask[:, bbox[0] + h:bbox[0] + bbox[2] - h,
        bbox[1] + w:bbox[1] + bbox[3] - w, :] = 1.
        return mask

    img_shape = FLAGS.img_shapes
    height = img_shape[0]
    width = img_shape[1]
    mask = tf.numpy_function(
        npmask,
        [bbox, height, width,
         FLAGS.max_delta_height, FLAGS.max_delta_width],
        tf.float32)
    mask.set_shape([1] + [height, width] + [1])
    return mask


def brush_stroke_mask(FLAGS, name='mask', _test_num=[0]):
    """Generate mask tensor from bbox.

    Returns:
        np.array: Mask array of shape [1, H, W, 1]
    """
    import numpy as np

    # Constants
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 5  # Original 12
    max_width = 18  # Original 40

    def generate_mask_circle(H, W, T):
        """
        Generate circular mask from bounding box.

        Args:
            H (int): Height of the mask
            W (int): Width of the mask
            T (int): Test number

        Returns:
            np.array: Mask array of shape (1, H, W, 1)
        """
        mask = np.zeros((1, H, W, 1), dtype=np.float32)  # Create empty mask

        # Circle parameters

        if (T < 40):
            cx, cy = T + 10, 10
        elif (40 <= T < 80):
            cx = T + 10 - 40
            cy = 20

        elif (80 <= T < 120):
            cx = T + 10 - 80
            cy = 30

        elif (120 <= T < 160):
            cx = T + 10 - 120
            cy = 40
        elif(160<=T<200):
            cx = T + 10 - 160
            cy = 50
        else:
            cx = T + 10 - 200
            cy = 60

        # Adjust radius based on test number

        radius = min(W, H) // 6

        # Draw circle
        for y in range(H):
            for x in range(W):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    mask[:, y, x, :] = 1

        return mask

    # Get input image shape
    img_shape = FLAGS.img_shapes
    height = img_shape[0]  # Image height
    width = img_shape[1]  # Image width

    # Generate mask
    mask = generate_mask_circle(height, width, _test_num[0])
    _test_num[0] += 1  # Increment test number for next call

    return mask  # Return the mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    to_shape = x.get_shape().as_list()[1:3]
    # align_corners=align_corners???
    x = tf.image.resize(mask, [to_shape[0], to_shape[1]], method='nearest')

    return x


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False, func='nearest', name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1] * scale, tf.int32),
                  tf.cast(xs[2] * scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
    if to_shape is None:
        x = tf.image.resize(x, new_xs)
    else:
        x = tf.image.resize(x, [to_shape[0], to_shape[1]], method=func)
    return x


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


@tf.function
def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    img = tf.numpy_function(flow_to_image, [flow], tf.float32)
    img.set_shape(flow.get_shape().as_list()[0:-1] + [3])
    img = img / 127.5 - 1.
    return img
