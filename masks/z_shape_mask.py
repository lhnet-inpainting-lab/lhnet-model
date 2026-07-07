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

    # 브러시 스트로크의 좌표를 임의로 생성합니다.
    vertex = [(W // 4, H // 4), (3 * W // 4, H // 4), (W // 4, 3 * H // 4), (3 * W // 4, 3 * H // 4)]
    
    # 생성한 점 주위에 브러시 스트로크를 그립니다.
    draw = ImageDraw.Draw(mask)
    width = 10  # 선의 두께
    draw.line(vertex, fill=1, width=width)
    for v in vertex:
        draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)

    # 마스크를 좌우로 뒤집을지, 상하로 뒤집을지 결정합니다.
    if np.random.normal() > 0:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
    mask = np.asarray(mask, np.float32)  # 마스크를 넘파이 어레이로 변환합니다.
    mask = np.reshape(mask, (1, H, W, 1))  # 마스크의 형상을 수정합니다.
    return mask

# 입력 이미지의 크기를 가져옵니다.
img_shape = FLAGS.img_shapes
height = img_shape[0]  # 이미지의 높이
width = img_shape[1]  # 이미지의 너비

# 마스크를 생성합니다.
mask = tf.numpy_function(generate_mask, [height, width], tf.float32)
mask.set_shape([1] + [height, width] + [1])  # 마스크의 형상을 설정합니다.

return mask  # 마스크를 반환합니다.
