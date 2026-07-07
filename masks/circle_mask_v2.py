import numpy as np

def generate_mask_circle(H, W):
    """
    바운딩 박스로부터 동그라미 모양의 마스크를 생성합니다.

    Args:
        H (int): 마스크의 높이
        W (int): 마스크의 너비

    Returns:
        np.array: 형상이 (1, H, W, 1)인 마스크 어레이
    """
    mask = np.zeros((1, H, W, 1), dtype=np.float32)  # 빈 마스크 생성

    # 원의 중심 좌표와 반지름 설정
    cx, cy = W // 2, H // 2
    radius = min(W, H) // 4  # 반지름 설정, 너비와 높이 중 작은 값을 기준으로 설정

    # 원 그리기
    for y in range(H):
        for x in range(W):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                mask[:, y, x, :] = 1

    # 마스크의 형태 수정
    return mask

# 입력 이미지의 크기를 가져옵니다.
img_shape = FLAGS.img_shapes
height = img_shape[0]  # 이미지의 높이
width = img_shape[1]  # 이미지의 너비

# 마스크 생성
mask = generate_mask_circle(height, width)
return mask  # 마스크 반환
