import numpy as np

def generate_mask_triangle(H, W):
    """
    바운딩 박스로부터 세모 모양의 마스크를 생성합니다.

    Args:
        H (int): 마스크의 높이
        W (int): 마스크의 너비

    Returns:
        np.array: 형상이 (1, H, W, 1)인 마스크 어레이
    """
    mask = np.zeros((1, H, W, 1), dtype=np.float32)  # 빈 마스크 생성

    # 세모의 좌표 설정
    x0, y0 = W // 2, H // 4
    x1, y1 = W // 4, 3 * H // 4
    x2, y2 = 3 * W // 4, 3 * H // 4

    # 세모 그리기
    for y in range(H):
        for x in range(W):
            if (x0 - x1) * (y - y1) - (y0 - y1) * (x - x1) > 0 and (x1 - x2) * (y - y2) - (y1 - y2) * (x - x2) > 0 and (x2 - x0) * (y - y0) - (y2 - y0) * (x - x0) > 0:
                mask[:, y, x, :] = 1

    # 마스크의 형태 수정
    return mask

# 입력 이미지의 크기를 가져옵니다.
img_shape = FLAGS.img_shapes
height = img_shape[0]  # 이미지의 높이
width = img_shape[1]  # 이미지의 너비

# 마스크 생성
mask = generate_mask_triangle(height, width)
return mask  # 마스크 반환
