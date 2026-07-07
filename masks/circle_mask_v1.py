import numpy as np

def generate_mask(H, W):
    """
    바운딩 박스로부터 브러시 스트로크 마스크를 생성합니다.

    Args:
        H (int): 마스크의 높이
        W (int): 마스크의 너비

    Returns:
        np.array: 형상이 (1, H, W, 1)인 마스크 어레이
    """
    mask = np.zeros((1, H, W, 1), dtype=np.float32)  # 빈 마스크 생성

    # 선을 그릴 좌표 설정
    x0, y0 = W // 4, H // 2
    x1, y1 = 3 * W // 4, H // 2
    thickness = 20  # 선의 두께

    # 선 안의 빈 공간 그리기
    mask[:, y0-thickness//2+1:y0+thickness//2, x0+1:x1, :] = 1

    # 마스크의 형태 수정
    return mask

# 입력 이미지의 크기를 가져옵니다.
img_shape = FLAGS.img_shapes
height = img_shape[0]  # 이미지의 높이
width = img_shape[1]  # 이미지의 너비

# 마스크 생성
mask = generate_mask(height, width)
return mask  # 마스크 반환
