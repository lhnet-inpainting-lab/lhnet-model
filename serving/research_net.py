"""연구모델(GeneratorMultiColumn) 추론용 네트워크 정의 — 2024 캡스톤 학습 코드 벤더링.

원본 net.py는 레포에서 삭제돼 이력(e4bc125)에만 남아 있었다. 여기서는 추론에 필요한
부분만 복원했다:
- GeneratorMultiColumn · gen_conv_block · gen_deconv_block: 이력 net.py
  (스냅샷에서 클래스 밖으로 밀려나 있던 call을 메서드로 복원)
- resize · resize_mask_like · contextual_attention: model/train/utils.py

학습·시각화 전용 의존(config/sn/Discriminator, offset flow 이미지 변환)은 제외했고,
contextual_attention은 flow 대신 None을 반환한다. 체크포인트(object-based restore)
호환을 위해 레이어 속성명·구조는 원본 그대로 유지한다 — stage3 후반에서 g140이 아닌
g40을 재사용하는 원본 코드의 특성도 학습 당시 그대로 보존했다.
"""

import tensorflow as tf


def resize(x, scale=2, to_shape=None, func='nearest'):
    xs = x.get_shape().as_list()
    new_xs = [int(xs[1] * scale), int(xs[2] * scale)]
    if to_shape is None:
        return tf.image.resize(x, new_xs)
    return tf.image.resize(x, [to_shape[0], to_shape[1]], method=func)


def resize_mask_like(mask, x):
    """mask를 x의 공간 해상도에 맞게 리사이즈한다."""
    to_shape = x.get_shape().as_list()[1:3]
    return tf.image.resize(mask, [to_shape[0], to_shape[1]], method='nearest')


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """전경 f를 배경 b의 패치로 재구성하는 contextual attention (원본 utils.py 이식)."""
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    kernel = 2 * rate
    raw_w = tf.image.extract_patches(
        b, [1, kernel, kernel, 1], [1, rate * stride, rate * stride, 1], [1, 1, 1, 1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])
    f = resize(f, scale=1. / rate, func='nearest')
    b = resize(b, to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)], func='nearest')
    if mask is not None:
        mask = resize(mask, scale=1. / rate, func='nearest')
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.image.extract_patches(
        b, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # b*k*k*c*hw
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.image.extract_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])
    m = m[0]
    mm = tf.cast(tf.math.equal(tf.math.reduce_mean(m, axis=[0, 1, 2], keepdims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    scale = softmax_scale
    k = fuse_k
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        wi = wi[0]
        wi_normed = wi / tf.math.maximum(
            tf.math.sqrt(tf.math.reduce_sum(tf.math.square(wi), axis=[0, 1, 2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

        # 큰 패치가 선택되도록 인접 점수를 융합
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

        yi *= mm
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm

        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(
            yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1, rate, rate, 1]) / 4.
        y.append(yi)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    # 원본은 offset flow 시각화 이미지를 함께 반환하지만 추론에는 쓰이지 않아 생략한다.
    return y, None


class gen_conv_block(tf.keras.layers.Layer):
    def __init__(self, filters, size, stride=1, dilation_rate=1, activation=tf.keras.activations.swish):
        super(gen_conv_block, self).__init__(name='')
        self.filters = filters
        self.activation = activation
        self.conv2d = tf.keras.layers.Conv2D(filters, size, stride, padding='same',
                                             dilation_rate=dilation_rate, activation=None)

    def call(self, input_tensor, training=False):
        x = self.conv2d(input_tensor)
        if self.filters == 3 or self.activation is None:
            return x

        x, y = tf.split(x, num_or_size_splits=2, axis=3)
        x = self.activation(x)
        y = tf.keras.activations.sigmoid(y)
        return x * y


class gen_deconv_block(tf.keras.layers.Layer):
    def __init__(self, filters, multi=0):
        super(gen_deconv_block, self).__init__(name='')
        self.multi = multi
        self.gen_conv_block = gen_conv_block(filters, 3, 1)

    def call(self, input_tensor, training=False):
        x = input_tensor
        if not self.multi:
            x = resize(x, func='nearest')
        return self.gen_conv_block(x)


class GeneratorMultiColumn(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cnum = 48

        self.g0_1 = gen_conv_block(cnum, 9, 1)
        self.g0_2 = gen_conv_block(2 * cnum, 9, 2)
        self.g0_3 = gen_conv_block(2 * cnum, 9, 1)
        self.g0_4 = gen_conv_block(4 * cnum, 9, 1)
        self.g0_5 = gen_conv_block(4 * cnum, 9, 1)
        self.g0_6 = gen_conv_block(4 * cnum, 9, 1)
        self.g0_7 = gen_conv_block(4 * cnum, 9, dilation_rate=2)
        self.g0_8 = gen_conv_block(4 * cnum, 9, dilation_rate=4)
        self.g0_9 = gen_conv_block(4 * cnum, 9, dilation_rate=8)
        self.g0_10 = gen_conv_block(4 * cnum, 9, dilation_rate=16)
        self.g0_11 = gen_conv_block(4 * cnum, 9, dilation_rate=32)
        self.g0_12 = gen_conv_block(4 * cnum, 7, 1)
        self.g0_13 = gen_conv_block(4 * cnum, 7, 1)

        self.g1_1 = gen_conv_block(cnum, 7, 1)
        self.g1_2 = gen_conv_block(2 * cnum, 7, 2)
        self.g1_3 = gen_conv_block(2 * cnum, 7, 1)
        self.g1_4 = gen_conv_block(4 * cnum, 7, 2)
        self.g1_5 = gen_conv_block(4 * cnum, 7, 1)
        self.g1_6 = gen_conv_block(4 * cnum, 7, 1)
        self.g1_7 = gen_conv_block(4 * cnum, 7, dilation_rate=2)
        self.g1_8 = gen_conv_block(4 * cnum, 7, dilation_rate=4)
        self.g1_9 = gen_conv_block(4 * cnum, 7, dilation_rate=8)
        self.g1_10 = gen_conv_block(4 * cnum, 7, dilation_rate=16)
        self.g1_11 = gen_conv_block(4 * cnum, 7, dilation_rate=32)
        self.g1_12 = gen_conv_block(4 * cnum, 7, 1)
        self.g1_13 = gen_conv_block(4 * cnum, 7, 1)
        self.g1_14 = gen_deconv_block(2 * cnum, multi=1)
        self.g1_15 = gen_conv_block(2 * cnum, 5, 1)

        self.g2_1 = gen_conv_block(cnum, 5, 1)
        self.g2_2 = gen_conv_block(2 * cnum, 5, 2)
        self.g2_3 = gen_conv_block(2 * cnum, 5, 1)
        self.g2_4 = gen_conv_block(4 * cnum, 5, 2)
        self.g2_5 = gen_conv_block(4 * cnum, 5, 1)
        self.g2_6 = gen_conv_block(4 * cnum, 5, 1)
        self.g2_7 = gen_conv_block(4 * cnum, 5, dilation_rate=2)
        self.g2_8 = gen_conv_block(4 * cnum, 5, dilation_rate=4)
        self.g2_9 = gen_conv_block(4 * cnum, 5, dilation_rate=8)
        self.g2_10 = gen_conv_block(4 * cnum, 5, dilation_rate=16)
        self.g2_11 = gen_conv_block(4 * cnum, 5, dilation_rate=32)
        self.g2_12 = gen_conv_block(4 * cnum, 5, 1)
        self.g2_13 = gen_conv_block(4 * cnum, 5, 1)
        self.g2_14 = gen_deconv_block(2 * cnum, multi=1)
        self.g2_15 = gen_conv_block(2 * cnum, 5, 1)

        self.g3_1 = gen_conv_block(cnum, 5, 1)
        self.g3_2 = gen_conv_block(2 * cnum, 3, 2)
        self.g3_3 = gen_conv_block(2 * cnum, 3, 1)
        self.g3_4 = gen_conv_block(4 * cnum, 3, 2)
        self.g3_5 = gen_conv_block(4 * cnum, 3, 1)
        self.g3_6 = gen_conv_block(4 * cnum, 3, 1)
        self.g3_7 = gen_conv_block(4 * cnum, 3, dilation_rate=2)
        self.g3_8 = gen_conv_block(4 * cnum, 3, dilation_rate=4)
        self.g3_9 = gen_conv_block(4 * cnum, 3, dilation_rate=8)
        self.g3_10 = gen_conv_block(4 * cnum, 3, dilation_rate=16)
        self.g3_11 = gen_conv_block(4 * cnum, 3, dilation_rate=32)
        self.g3_12 = gen_conv_block(4 * cnum, 3)
        self.g3_13 = gen_conv_block(4 * cnum, 3, 1)
        self.g3_14 = gen_deconv_block(2 * cnum, multi=1)
        self.g3_15 = gen_conv_block(2 * cnum, 3, 1)
        self.g3_16 = gen_deconv_block(cnum, multi=1)
        self.g3_17 = gen_conv_block(cnum // 2, 3, 1)

        self.m1 = gen_conv_block(cnum // 2, 3, 1)
        self.m2 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation=None)

        # conv_branch1
        self.g18 = gen_conv_block(cnum, 5, 1)
        self.g19 = gen_conv_block(cnum, 3, 2)
        self.g20 = gen_conv_block(2 * cnum, 3, 1)
        self.g21 = gen_conv_block(2 * cnum, 3, 2)
        self.g22 = gen_conv_block(4 * cnum, 3, 1)
        self.g23 = gen_conv_block(4 * cnum, 3, 1)
        self.g24 = gen_conv_block(4 * cnum, 3, dilation_rate=2)
        self.g25 = gen_conv_block(4 * cnum, 3, dilation_rate=4)
        self.g26 = gen_conv_block(4 * cnum, 3, dilation_rate=8)
        self.g27 = gen_conv_block(4 * cnum, 3, dilation_rate=16)

        # attention branch / conv_blocks1
        self.g28 = gen_conv_block(cnum, 5, 1)
        self.g29 = gen_conv_block(cnum, 3, 2)
        self.g30 = gen_conv_block(2 * cnum, 3, 1)
        self.g31 = gen_conv_block(4 * cnum, 3, 2)
        self.g32 = gen_conv_block(4 * cnum, 3, 1)
        self.g33 = gen_conv_block(4 * cnum, 3, 1, activation=tf.keras.activations.relu)

        # before pm1
        self.g34 = gen_conv_block(4 * cnum, 3, 1)
        self.g35 = gen_conv_block(4 * cnum, 3, 1)

        # before conv3 blocks
        self.m3 = gen_conv_block(8 * cnum, 3, 1)

        # conv3 blocks
        self.g36 = gen_conv_block(4 * cnum, 3, 1)
        self.g37 = gen_conv_block(4 * cnum, 3, 1)
        self.g38 = gen_deconv_block(2 * cnum)
        self.g39 = gen_conv_block(2 * cnum, 3, 1)
        self.g40 = gen_deconv_block(cnum)
        self.g41 = gen_conv_block(cnum // 2, 3, 1)
        self.g42 = gen_conv_block(3, 3, 1, activation=tf.keras.activations.tanh)

        # stage3 conv_branch
        self.g118 = gen_conv_block(cnum, 5, 1)
        self.g119 = gen_conv_block(cnum, 3, 2)
        self.g120 = gen_conv_block(2 * cnum, 3, 1)
        self.g121 = gen_conv_block(2 * cnum, 3, 1)
        self.g122 = gen_conv_block(4 * cnum, 3, 1)
        self.g123 = gen_conv_block(4 * cnum, 3, 1)
        self.g124 = gen_conv_block(4 * cnum, 3, dilation_rate=2)
        self.g125 = gen_conv_block(4 * cnum, 3, dilation_rate=4)
        self.g126 = gen_conv_block(4 * cnum, 3, dilation_rate=8)
        self.g127 = gen_conv_block(4 * cnum, 3, dilation_rate=16)

        # stage3 attention branch / conv_blocks2
        self.g128 = gen_conv_block(cnum, 5, 1)
        self.g129 = gen_conv_block(cnum, 3, 2)
        self.g130 = gen_conv_block(2 * cnum, 3, 1)
        self.g131 = gen_conv_block(4 * cnum, 3, 1)
        self.g132 = gen_conv_block(4 * cnum, 3, 1)
        self.g133 = gen_conv_block(4 * cnum, 3, 1, activation=tf.keras.activations.relu)

        # stage3 before pm2
        self.g134 = gen_conv_block(4 * cnum, 3, 1)
        self.g135 = gen_conv_block(4 * cnum, 3, 1)

        # stage3 before conv4 blocks
        self.m4 = gen_conv_block(8 * cnum, 3, 1)

        # stage3 conv4 blocks
        self.g136 = gen_conv_block(4 * cnum, 3, 1)
        self.g137 = gen_conv_block(4 * cnum, 3, 1)
        self.g138 = gen_conv_block(2 * cnum, 3, 1)
        self.g139 = gen_conv_block(2 * cnum, 3, 1)
        self.g140 = gen_deconv_block(cnum)
        self.g141 = gen_conv_block(cnum // 2, 3, 1)
        self.g142 = gen_conv_block(3, 3, 1, activation=tf.keras.activations.tanh)

    def call(self, x, mask):
        xin = x
        offset_flow1 = None
        offset_flow2 = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]

        x_noise = tf.keras.layers.GaussianNoise(stddev=0.1)(x)
        x_w_mask = tf.concat([x_noise, ones_x, ones_x * mask], axis=3)
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]

        # STAGE 1 — BRANCH 0
        x = self.g0_1(x_w_mask)
        x = self.g0_2(x)
        x = self.g0_3(x)
        x = self.g0_4(x)
        x = self.g0_5(x)
        x = self.g0_6(x)
        mask_s0 = resize_mask_like(mask, x)
        x = self.g0_7(x)
        x = self.g0_8(x)
        x = self.g0_9(x)
        x = self.g0_10(x)
        x = self.g0_11(x)
        x = self.g0_12(x)
        x = self.g0_13(x)
        x_b0 = tf.image.resize(x, [xh, xw], method='bilinear')

        # BRANCH 1
        x = self.g1_1(x_w_mask)
        x = self.g1_2(x)
        x = self.g1_3(x)
        x = self.g1_4(x)
        x = self.g1_5(x)
        x = self.g1_6(x)
        mask_s1 = resize_mask_like(mask, x)
        x = self.g1_7(x)
        x = self.g1_8(x)
        x = self.g1_9(x)
        x = self.g1_10(x)
        x = self.g1_11(x)
        x = self.g1_12(x)
        x = self.g1_13(x)
        x = tf.image.resize(x, [xh // 2, xw // 2], method='bilinear')
        x = self.g1_14(x)
        x = self.g1_15(x)
        x_b1 = tf.image.resize(x, [xh, xw], method='bilinear')

        # BRANCH 2
        x = self.g2_1(x_w_mask)
        x = self.g2_2(x)
        x = self.g2_3(x)
        x = self.g2_4(x)
        x = self.g2_5(x)
        x = self.g2_6(x)
        x = self.g2_7(x)
        x = self.g2_8(x)
        x = self.g2_9(x)
        x = self.g2_10(x)
        x = self.g2_11(x)
        x = self.g2_12(x)
        x = self.g2_13(x)
        x = tf.image.resize(x, [xh // 2, xw // 2], method='bilinear')
        x = self.g2_14(x)
        x = self.g2_15(x)
        x_b2 = tf.image.resize(x, [xh, xw], method='bilinear')

        # BRANCH 3
        x = self.g3_1(x_w_mask)
        x = self.g3_2(x)
        x = self.g3_3(x)
        x = self.g3_4(x)
        x = self.g3_5(x)
        x = self.g3_6(x)
        x = self.g3_7(x)
        x = self.g3_8(x)
        x = self.g3_9(x)
        x = self.g3_10(x)
        x = self.g3_11(x)
        x = self.g3_12(x)
        x = self.g3_13(x)
        x = tf.image.resize(x, [xh // 2, xw // 2], method='nearest')
        x = self.g3_14(x)
        x = self.g3_15(x)
        x = tf.image.resize(x, [xh, xw], method='nearest')
        x = self.g3_16(x)
        x_b3 = self.g3_17(x)

        x_merge = tf.concat([x_b0, x_b1, x_b2, x_b3], axis=3)
        x = self.m1(x_merge)
        x = self.m2(x)
        x = tf.clip_by_value(x, -1., 1.)
        x_stage1 = x

        # STAGE 2
        x = x * mask + xin[:, :, :, 0:3] * (1. - mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        xnow1 = x

        x = self.g18(xnow1)
        x = self.g19(x)
        x = self.g20(x)
        x = self.g21(x)
        x = self.g22(x)
        x = self.g23(x)
        x = self.g24(x)
        x = self.g25(x)
        x = self.g26(x)
        x = self.g27(x)
        x_hallu1 = x

        x = self.g28(xnow1)
        x = self.g29(x)
        x = self.g30(x)
        x = self.g31(x)
        x = self.g32(x)
        x = self.g33(x)
        x, offset_flow1 = contextual_attention(x, x, mask_s1, 3, 1, rate=2)
        x = self.g34(x)
        x = self.g35(x)
        pm1 = x
        x = tf.concat([x_hallu1, pm1], axis=3)
        x = self.m3(x)
        x = tf.clip_by_value(x, -1., 1.)

        x = self.g36(x)
        x = self.g37(x)
        x = self.g38(x)
        x = self.g39(x)
        x = self.g40(x)
        x = self.g41(x)
        x = self.g42(x)
        x_stage2 = x

        # STAGE 3
        x = x * mask + xin[:, :, :, 0:3] * (1. - mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        xnow2 = x

        x = self.g118(xnow2)
        x = self.g119(x)
        x = self.g120(x)
        x = self.g121(x)
        x = self.g122(x)
        x = self.g123(x)
        x = self.g124(x)
        x = self.g125(x)
        x = self.g126(x)
        x = self.g127(x)
        x_hallu2 = x

        x = self.g128(xnow2)
        x = self.g129(x)
        x = self.g130(x)
        x = self.g131(x)
        x = self.g132(x)
        x = self.g133(x)
        x, offset_flow2 = contextual_attention(x, x, mask_s0, 3, 1, rate=2)
        x = self.g134(x)
        x = self.g135(x)
        pm2 = x

        x = tf.concat([x_hallu2, pm2], axis=3)
        x = self.m4(x)
        x = tf.clip_by_value(x, -1., 1.)

        x = self.g136(x)
        x = self.g137(x)
        x = self.g138(x)
        x = self.g139(x)
        x = self.g40(x)  # 원본 그대로: 학습 당시에도 g140 대신 g40이 쓰였다
        x = self.g141(x)
        x = self.g142(x)
        x_stage3 = x

        return x_stage1, x_stage2, x_stage3, offset_flow1, offset_flow2
