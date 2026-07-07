import tensorflow as tf
import numpy as np
import math
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import datetime

from config import Config

"""
이 코드는 TensorFlow를 사용하여 이미지 인페인팅을 위한 마스크를 생성하는 기능을 구현합니다.
이미지에서 일정 영역을 무작위로 선택하고, 선택한 영역에 대한 정규 마스크와 브러시 스트로크 마스크를 생성하여 결합하여 최종 마스크를 반환합니다.
정규 마스크는 이미지에서 일정 영역을 무작위로 선택하여 해당 영역을 채우는 마스크입니다. 이것은 주로 직사각형 모양으로 선택되며, 선택된 영역을 가리키는 픽셀은 1로, 그렇지 않은 픽셀은 0으로 표시됩니다.
스트로크 마스크는 브러시 스트로크 효과를 시뮬레이션하여 마스크를 만듭니다. 이것은 더 자연스러운 결과를 얻기 위해 선택된 영역 주변에 흐릿한 효과를 부여하는 데 사용됩니다. 
이 마스크는 선택된 영역에 대한 부분적인 투명도를 가지며, 마스크의 값은 0에서 1 사이의 연속적인 값으로 표현됩니다.

"""
# Config 클래스로부터 설정을 불러옵니다.
FLAGS = Config('./inpaint.yml')
img_shape = FLAGS.img_shapes
IMG_HEIGHT = img_shape[0]
IMG_WIDTH = img_shape[1]


# psnr코드
import numpy


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


# 이미지를 로드하는 함수입니다.
def load(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img)
    return tf.cast(img, tf.float32)


# 이미지를 정규화하는 함수입니다.
def normalize(img):
    return (img / 127.5) - 1.


# 학습용 이미지를 로드하는 함수입니다.
def load_image_train(img):
    img = load(img)
    img = resize_pipeline(img, IMG_HEIGHT, IMG_WIDTH)
    return normalize(img)


# 이미지를 리사이즈하는 파이프라인입니다.
def resize_pipeline(img, height, width):
    return tf.image.resize(img, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


# CSV 파일을 읽어들이는 함수입니다.
def CSV_reader(input):
    import re
    input = [i.split('tf.Tensor(')[1].split(', shape')[0] for i in input]
    return tf.strings.to_number(input)


# 마스크를 생성하는 함수입니다.
def create_mask(FLAGS):
    # 무작위 바운딩 박스를 생성합니다.
    bbox = random_bbox(FLAGS)
    # 정규 마스크를 생성합니다.
    # bbox2mask 함수는 주어진 bounding box를 사용하여 정규 마스크를 생성하는 함수입니다. 이 함수는 입력으로 FLAGS, bbox 및 선택적으로 name을 받습니다.
    # FLAGS는 설정을 포함하는 객체이며, bbox는 bounding box를 나타내는 변수입니다
    regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')

    # 불규칙 마스크를 생성합니다.
    irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
    # 정규 마스크와 불규칙 마스크를 합칩니다.
    mask = tf.cast(
        tf.math.logical_or(
            tf.cast(irregular_mask, tf.bool),
            tf.cast(regular_mask, tf.bool),
        ),
        tf.float32
    )
    return mask


def generate_images(input, generator, training=True, url=False, num_epoch=0):
    """
    입력 이미지를 사용하여 이미지 인페인팅 모델에서 예측된 결과를 생성하고 시각화하는 함수.

    Args:
        input (tf.Tensor): 입력 이미지.
        generator (callable): 인페인팅 모델.
        training (bool): 이미지가 훈련용인지 테스트용인지 여부를 나타내는 플래그. 기본값은 True.
        url (bool): 이미지를 반환할지 파일로 저장할지 여부를 나타내는 플래그. 기본값은 False.
        num_epoch (int): 현재 에포크 수. 파일로 저장할 때 사용됨. 기본값은 0.

    Returns:
        tuple or None: url이 True일 경우 (batch_incomplete, batch_complete) 튜플 반환, 그렇지 않으면 None 반환.
    """
    # 입력 이미지에 대한 마스크 생성
    mask = create_mask(FLAGS)
    # 입력 이미지에 마스크를 적용하여 부분 입력 이미지 생성
    batch_incomplete = input * (1. - mask)
    # 부분 입력 이미지를 모델에 전달하여 예측 결과 생성
    stage1, stage2, offset_flow = generator(batch_incomplete, mask, training=training)

    plt.figure(figsize=(30, 30))

    batch_predict = stage2
    # 마스크를 적용한 예측 결과와 마스크가 적용되지 않은 부분 입력 이미지를 결합하여 완전한 이미지 생성
    batch_complete = batch_predict * mask + batch_incomplete * (1 - mask)

    # input_mask vs stage2_mask
    '''
    input_mask= input[0] * mask[0]
    stage2_mask = batch_predict[0] * mask[0]
    '''
    input_mask = input[0] * mask
    stage2_mask = batch_predict[0] * mask

    # psnr코드2
    # input vs stage2
    cal_psnr = psnr(input[0], batch_predict[0])
    # input vs inpainted image
    cal_psnr1 = psnr(input[0], batch_complete[0])
    # input mask vs stage2 mask

    cal_psnr2 = psnr(input[0] * mask, batch_predict[0] * mask)
    cal_psnr3 = psnr(input[0] * mask, batch_complete[0] * mask)
    cal_psnr4 = psnr(input[0] * mask, batch_predict * mask)
    cal_psnr5 = psnr(input[0] * mask, batch_complete[0] * (1 - mask) + batch_predict[0] * mask)

    print('PSNR: input vs stage2 = %.4f' % cal_psnr)
    print('PSNR: input vs inpainted= %.4f' % cal_psnr1)

    print('PSNR: input_mask vs stage2_mask  =%.4f' % cal_psnr4)

    print('PSNR: input_mask vs stage2_mask(test)=%.4f' % cal_psnr5)
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

    # 시각화할 이미지 목록 및 제목 설정
    display_list = [input[0], batch_incomplete[0], stage1[0], stage2[0], batch_complete[0], offset_flow[0]]
    title = ['Input Image', 'Input With Mask', 'Stage 1', 'Stage 2', 'Inpainted Image', 'Offset Flow']

    if not url:
        # 이미지를 그래픽으로 출력
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            title_obj = plt.title(title[i])
            plt.setp(title_obj, color='y')  # 제목 색상을 노란색으로 설정
            plt.axis('off')
            # 픽셀 값을 [0, 1] 사이로 조정하여 이미지 플로팅
            plt.imshow(display_list[i] * 0.5 + 0.5)
        if training:
            plt.savefig(f"./images_examples/test_example_{num_epoch}.png")
        else:
            # 테스트 시에는 현재 시간을 사용하여 파일명에 시간 정보 추가하여 저장
            plt.savefig(f"./images_examples/infer_test_example_{num_epoch}__" + datetime.datetime.now().strftime(
                "%H%M%S%f") + ".png")
    else:
        # 이미지를 반환
        return batch_incomplete[0], batch_complete[0]


def plot_history(g_total_h, g_hinge_h, g_l1_h, d_h, num_epoch, training=True):
    """
    학습 중에 생성된 손실값을 시각화하여 그래프로 표시하는 함수.

    Args:
        g_total_h (list): 생성자의 총 손실값 리스트.
        g_hinge_h (list): 생성자의 힌지 손실값 리스트.
        g_l1_h (list): 생성자의 L1 손실값 리스트.
        d_h (list): 판별자의 손실값 리스트.
        num_epoch (int): 현재 에포크 수. 파일로 저장할 때 사용됨.
        training (bool): 훈련 중인지 테스트 중인지를 나타내는 플래그. 기본값은 True.
    """
    # 그래프 크기 설정
    plt.figure(figsize=(20, 10))

    # 총 생성자 손실 그래프
    plt.subplot(4, 1, 1)
    plt.plot(g_total_h, label='total_gen_loss')
    plt.legend()

    # 생성자 힌지 손실 그래프
    plt.subplot(4, 1, 2)
    plt.plot(g_hinge_h, label='gen_hinge_loss')
    plt.legend()

    # 생성자 L1 손실 그래프
    plt.subplot(4, 1, 3)
    plt.plot(g_l1_h, label='gen_l1_loss')
    plt.legend()

    # 판별자 손실 그래프
    plt.subplot(4, 1, 4)
    plt.plot(d_h, label='dis_loss')
    plt.legend()

    # 그래프를 파일로 저장
    if training:
        plt.savefig(f"./images_loss/plot_loss_{num_epoch}.png")
    else:
        plt.savefig(f"./images_loss/infer_plot_loss_{num_epoch}.png")

    # 그래프 출력 초기화
    plt.clf()
    plt.close()


# COMPUTATIONS
def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., training=True,
                         fuse=True):
    """
    Contextual Attention 메커니즘을 사용하여 입력 특성 맵에서 배경 이미지로부터 정보를 가져와 예측 결과를 생성하는 함수.

    Args:
        f (tf.Tensor): 특성 맵.
        b (tf.Tensor): 배경 이미지.
        mask (tf.Tensor): 마스크. 기본값은 None.
        ksize (int): 커널 크기. 기본값은 3.
        stride (int): 스트라이드. 기본값은 1.
        rate (int): 비율. 기본값은 1.
        fuse_k (int): 퓨즈 케이. 기본값은 3.
        softmax_scale (float): 소프트맥스 스케일. 기본값은 10.0.
        training (bool): 훈련 중인지 테스트 중인지를 나타내는 플래그. 기본값은 True.
        fuse (bool): 퓨즈 여부를 나타내는 플래그. 기본값은 True.

    Returns:
        tf.Tensor, tf.Tensor: 예측된 결과 및 시각화된 플로우 이미지.
    """
    """
    이 부분은 텐서의 형태를 조사하는 것입니다. 
    tf.shape(f)는 텐서 f의 현재 형태를 반환하며, 
    f.get_shape().as_list()는 텐서의 형태를 리스트로 반환합니다. 
    raw_int_fs[0] = 1 및 raw_int_bs[0] = 1은 첫 번째 차원의 크기를 1로 설정하는 작업입니다.
    """
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # raw_int_fs[0] = 1
    # raw_int_bs[0] = 1
    # print("raw_int_bs" , raw_int_bs)

    # 커널 사이즈 및 raw_w 설정

    """
    이 부분은 배경 이미지에서 패치를 추출하는 과정입니다. 패치는 입력 이미지에서 일부 영역을 잘라내어 추출한 작은 부분 이미지를 나타냅니다. 
    이 과정은 주어진 rate 및 stride로 정의된 스케일링 및 스트라이딩을 사용하여 수행됩니다.

    kernel = 2 * rate: 패치의 크기를 결정하는 커널 크기를 설정합니다. 이 값은 rate에 비례하여 결정됩니다.
    tf.image.extract_patches(): 배경 이미지 b에서 패치를 추출합니다. 이 함수는 주어진 커널 크기와 스트라이드에 따라 이미지에서 패치를 추출합니다.
    tf.reshape(): 추출된 패치를 원하는 형태로 재구성합니다. 이 경우 패치를 텐서의 리스트로 변환하여 저장합니다.
    tf.transpose(): 재구성된 패치의 차원을 변경합니다. 이 코드는 패치를 텐서의 마지막 차원으로 이동하여 텐서를 더 쉽게 처리할 수 있도록 합니다.
    이 과정을 통해 배경 이미지에서 추출된 패치가 필요한 형태로 구성되어 있습니다.


    """
    kernel = 2 * rate
    raw_w = tf.image.extract_patches(
        b, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])

    # 특성 맵 및 배경 이미지 크기 조정
    """
    이 부분은 입력된 특성 맵 및 배경 이미지의 크기를 조정하는 과정입니다. 
    이것은 후속 과정에서 컨텍스트 간의 상호 작용을 적용하고 연산을 수행하기 위해 필요합니다.
    f = resize(f, scale=1. / rate, func='nearest'): 특성 맵 f를 크기 비율 rate에 따라 조정합니다. 
    이 경우, 입력 이미지를 1/rate 비율로 축소합니다. 이를 통해 이미지를 줄이고 컨텍스트 간의 상호 작용을 더 잘 파악할 수 있습니다.
    b = resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)], func='nearest'): 배경 이미지 b를 주어진 모양으로 조정합니다.
    raw_int_bs[1] 및 raw_int_bs[2]는 배경 이미지의 높이와 너비를 나타내는 값입니다. rate에 따라 배경 이미지의 크기를 조정합니다. 
    이렇게 함으로써 배경 이미지를 입력 이미지와 동일한 크기로 맞출 수 있습니다.

    이 과정을 통해 입력된 특성 맵 및 배경 이미지가 올바른 크기로 조정되어 컨텍스트 간의 상호 작용을 적용할 수 있게 됩니다.
    """
    f = resize(f, scale=1. / rate, func='nearest')
    b = resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
               func='nearest')  # https://github.com/tensorflow/tensorflow/issues/11651

    # 마스크 크기를 조정합니다.
    if mask is not None:
        mask = resize(mask, scale=1. / rate, func='nearest')

    # 특성 맵의 형태 및 배경 이미지의 형태를 가져옵니다.
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()

    # 특성 맵을 여러 그룹으로 분할합니다.
    f_groups = tf.split(f, int_fs[0], axis=0)

    # 배경 이미지를 패치로 추출합니다.
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.image.extract_patches(
        b, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # b*k*k*c*hw 형태로 변환합니다.

    """
    이 코드 블록은 주어진 마스크에서 패치를 추출하고, 
    패치들의 평균값을 계산하여 마스크를 생성합니다. 
    그리고 관련된 텐서를 그룹으로 분할하고 결과를 저장할 빈 리스트를 생성합니다.
    마지막으로 소프트맥스 스케일 및 퓨즈 크기를 설정합니다.
    """
    # 마스크 처리
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])  # 만약 마스크가 없다면, 크기가 1xbs[1]xbs[2]x1인 0으로 채워진 텐서를 생성합니다.

    # 마스크로부터 패치를 추출합니다.
    m = tf.image.extract_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])  # 패치를 원하는 형태로 재구성합니다.
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # 패치를 b*k*k*c*hw 형태로 변환합니다.
    # 이렇게 하면 계산량을 줄이고, 메모리를 절약할 수 있습니다. 따라서 첫 번째 요소만 선택하여 처리하고 있습니다.
    # 마스크 자체를 패치로 변환하고 있습니다.
    m = m[0]  # 첫 번째 요소만 선택하여 처리합니다.

    # 일반적으로 이미지 처리에서, 마스크가 0이 아닌 값을 가지는 영역은 주로 관심 있는 영역을 나타내고, 0인 값을 가지는 영역은 관심 없는 영역을 나타냅니다. 따라서 패치들의 평균값이 0인 경우, 해당 패치 영역은 배경이라고 가정할 수 있습니다.
    # 이러한 배경 영역을 마스크로 사용하여 후속 처리 단계에서 관심 있는 영역에 집중할 수 있습니다.
    # 패치들의 평균값을 계산하고 0과 비교하여 마스크를 생성합니다.
    mm = tf.cast(tf.math.equal(tf.math.reduce_mean(m, axis=[0, 1, 2], keepdims=True), 0.), tf.float32)

    # 배경 이미지와 관련된 텐서를 그룹으로 분할합니다.
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)

    # 결과를 저장할 빈 리스트를 생성합니다.
    y = []
    offsets = []

    # 소프트맥스 스케일 및 퓨즈 크기를 설정합니다.
    scale = softmax_scale
    k = fuse_k
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])  # 퓨즈 가중치를 설정합니다.

    """
    이 코드 블록은 주어진 특성 맵과 배경 이미지의 패치를 비교하여 어텐션 매커니즘을 수행하는 부분입니다. 어텐션 매커니즘은 특정 위치에 주목하도록 네트워크를 가르치는 기술입니다.

    여기서는 먼저, 주어진 배경 이미지의 패치들과 특성 맵을 비교하여 어텐션을 수행합니다. 패치들은 배경 이미지에서 추출되며, 특성 맵은 네트워크의 출력입니다. 이들을 비교하여 어텐션 스코어를 계산하고, 이를 소프트맥스 함수를 통해 확률 분포로 변환합니다.

    그런 다음, 어텐션 스코어를 기반으로 최대값 위치를 찾아 오프셋을 계산합니다. 이렇게 계산된 오프셋을 사용하여 패치를 특성 맵에 붙여넣습니다.

    마지막으로, 이러한 작업을 통해 얻은 결과와 옵티컬 플로우를 반환합니다. 옵티컬 플로우는 오프셋을 시각화한 것으로, 어텐션 매커니즘의 결과를 이해하고 디버깅하는 데 도움이 됩니다.

    """
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # 패치를 비교하기 위한 컨볼루션 수행
        wi = wi[0]  # 첫 번째 패치를 선택합니다.
        # 패치 정규화
        wi_normed = wi / tf.math.maximum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

        # 큰 패치를 장려하기 위한 퓨즈 스코어를 위한 컨볼루션 구현

        """
        이 부분은 fuse 매개변수에 따라 수행되는 작업입니다. fuse 매개변수가 True로 설정되면, 다음과 같은 과정이 수행됩니다:

        yi 텐서의 형태를 재구성하여 새로운 형태의 텐서를 만듭니다. 이때, 첫 번째 차원은 1이 되고, 나머지 차원은 특성 맵의 크기와 배경 이미지의 크기를 결합하여 새로운 형태를 만듭니다.
        이후, tf.nn.conv2d 함수를 사용하여 새로운 형태의 yi 텐서에 fuse_weight 필터를 적용합니다. 이 과정은 컨볼루션 연산을 수행합니다.
        다시 한번 yi 텐서의 형태를 재구성합니다. 이 과정에서는 특성 맵과 배경 이미지의 차원을 다시 복구합니다.
        텐서의 차원을 변경하고 순서를 변경하는 여러 단계를 거쳐 다시 yi 텐서를 재구성합니다.
        이러한 과정은 컨볼루션 연산을 통해 대상 특성을 강화하고 큰 패치에 더 많은 주목을 받도록 하는 것을 목적으로 합니다. 이는 어텐션 메커니즘의 성능을 향상시키고 더 좋은 결과를 얻기 위한 것입니다.
        """
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

        # 소프트맥스로 매칭
        # 어텐션 매커니즘에 대한 추가 처리 단계:
        # 1. 소프트맥스로 매칭:
        #    - 주어진 입력 `yi`에 마스크 `mm`을 적용합니다.
        #    - 소프트맥스 함수를 사용하여 확률 분포로 변환하여 각 위치의 중요도를 나타내는 어텐션 맵을 생성합니다.
        # 2. 마스크 적용:
        #    - 어텐션 맵에 마스크 `mm`을 다시 적용합니다.
        #    - 마스크가 있는 위치에서의 중요도를 0으로 만들어 해당 위치를 무시합니다.
        # 3. 스케일링:
        #    - 어텐션 맵의 값을 조정하기 위해 `scale` 변수를 사용합니다.
        #    - 이 단계는 어텐션 맵을 스케일링하여 결과를 보다 잘 조절합니다.

        yi *= mm  # 마스크를 적용합니다.
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm  # 마스크를 적용합니다.

        # 최대값 위치를 찾아서 오프셋을 계산합니다.
        # - 어텐션 맵에서 각 위치에서의 최대값을 찾습니다.
        # - `tf.math.argmax` 함수를 사용하여 최대값의 인덱스를 찾습니다.
        # - 인덱스를 2차원 형태로 변환하여 각 위치에 대한 오프셋을 계산합니다.
        offset = tf.math.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)

        # 패치 붙여넣기를 위한 디컨볼루션
        wi_center = raw_wi[0]  # 패치의 중심을 선택합니다.
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0),
                                    strides=[1, rate, rate, 1]) / 4.  # 패치 붙여넣기를 수행합니다.
        y.append(yi)  # 결과를 리스트에 추가합니다.
        offsets.append(offset)  # 오프셋을 리스트에 추가합니다.

    # 결과와 오프셋을 결합합니다:
    # - 각 패치에 대한 결과를 모아 하나의 텐서로 합칩니다.
    # - 결과 텐서의 형태를 설정합니다.
    # - 각 패치에 대한 오프셋을 모아 하나의 텐서로 합칩니다.
    # - 오프셋 텐서의 형태를 설정합니다.

    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])

    # 옵티컬 플로우를 시각화합니다:
    # - 각 위치에 대한 오프셋을 시각화하기 위해 현재 위치를 뺍니다.
    # - 오프셋을 이미지로 변환하여 옵티컬 플로우를 생성합니다.
    # - 만약 크기가 1이 아니라면, 옵티컬 플로우의 크기를 조정합니다.
    # - 최종적으로 결과와 옵티컬 플로우를 반환합니다.

    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)  # 현재 위치를 뺍니다.
    flow = flow_to_image_tf(offsets)  # 옵티컬 플로우를 이미지로 변환합니다.
    if rate != 1:
        flow = resize(flow, scale=rate, func='bilinear')  # 크기를 조정합니다.

    return y, flow  # 결과 및 옵티컬 플로우를 반환합니다.


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


def brush_stroke_mask(FLAGS, name='mask'):
    """
    바운딩 박스로부터 마스크 텐서를 생성합니다.

    Args:
        FLAGS: 마스크 생성 과정을 제어하는 플래그 또는 설정.
        name (str): 생성된 마스크 텐서의 이름.

    Returns:
        tf.Tensor: 형상이 [1, H, W, 1]인 출력 마스크 텐서.
    """
    # 사용할 변수들을 미리 선언합니다.
    min_num_vertex = 4  # 최소 점의 수
    max_num_vertex = 12  # 최대 점의 수
    mean_angle = 2 * math.pi / 5  # 평균 각도
    angle_range = 2 * math.pi / 15  # 각도 범위
    min_width = 5  # 최소 선의 너비 (원본 값은 12)
    max_width = 18  # 최대 선의 너비 (원본 값은 40)

    def generate_mask(H, W):
        """
        바운딩 박스로부터 브러시 스트로크 마스크를 생성합니다.

        Args:
            H (int): 마스크의 높이
            W (int): 마스크의 너비

        Returns:
            np.array: 형상이 (1, H, W, 1)인 마스크 어레이
        """
        average_radius = math.sqrt(H * H + W * W) / 8  # 평균 반지름
        mask = Image.new('L', (W, H), 0)  # 마스크 이미지를 생성합니다.

        # 고정된 위치에 브러시 스트로크를 그림
        fixed_x = W // 2
        fixed_y = H // 2
        vetex = [[fixed_x, fixed_y]]

        # 고정된 값을 사용하여 마스크를 생성합니다.
        num_vertex = 8  # 고정된 값으로 점의 수 설정
        mean_angle = 2 * math.pi / 5  # 고정된 값으로 평균 각도 설정
        angle_range = 2 * math.pi / 15  # 고정된 값으로 각도 범위 설정
        min_width = 5  # 고정된 값으로 최소 선의 너비 설정
        max_width = 18  # 고정된 값으로 최대 선의 너비 설정

        # 랜덤한 각도 대신 고정된 값을 사용합니다.
        angles = [mean_angle - angle_range, mean_angle + angle_range,
                  mean_angle - angle_range, mean_angle + angle_range,
                  mean_angle - angle_range, mean_angle + angle_range,
                  mean_angle - angle_range, mean_angle + angle_range]

        h, w = mask.size  # 마스크 이미지의 크기
        # 랜덤한 점을 생성하지 않고, 고정된 위치를 사용합니다.
        vertex = [(fixed_x, fixed_y)]

        # 생성한 점 주위에 브러시 스트로크를 그립니다.
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        # 마스크에 선을 그리고, 그 주위에 원을 그립니다.
        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))  # 선의 두께
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2,
                          v[1] - width // 2,
                          v[0] + width // 2,
                          v[1] + width // 2),
                         fill=1)

        # 랜덤하게 이미지를 좌우로 뒤집습니다.
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        # 랜덤하게 이미지를 상하로 뒤집습니다.
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)  # 마스크를 넘파이 어레이로 변환합니다.
        mask = np.reshape(mask, (1, H, W, 1))  # 마스크의 형상을 수정합니다.
        return mask

    # 입력 이미지의 크기를 가져옵니다.
    img_shape = FLAGS.img_shapes
    height = img_shape[0]  # 이미지의 높이
    width = img_shape[1]  # 이미지의 너비
    # 마스크를 생성합니다.
    mask = tf.numpy_function(
        generate_mask,
        [height, width],
        tf.float32)
    mask.set_shape([1] + [height, width] + [1])  # 마스크의 형상을 설정합니다.
    return mask  # 마스크를 반환합니다.


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
