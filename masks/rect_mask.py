def brush_stroke_mask(FLAGS, name='mask'):
    test_num = 0

    """Generate mask tensor from bbox.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """

    #Εδώ έβαλα μικρότερα τα max_width και min_width γιατί οι εικόνες 
    #όταν το τρέχω με 64X64Χ3 είναι πολύ μικρές για μία τέτοια μάσκα.

    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 5                     #Original 12
    max_width = 18                    #Original 40
    import numpy as np

    def generate_mask(H, W, T):
        """
        바운딩 박스로부터 브러시 스트로크 마스크를 생성합니다.

        Args:
            H (int): 마스크의 높이
            W (int): 마스크의 너비

        Returns:
            np.array: 형상이 (1, H, W, 1)인 마스크 어레이
        """
        mask = np.zeros((1, H, W, 1), dtype=np.float32)  # 빈 마스크 생성

        # 직사각형의 중심 좌표와 크기 설정

        # 크기 설정
        cx, cy = W // 2, H // 2  # 이미지의 중심 좌표
        if(T<5):
            rect_width = W // 4  # 직사각형의 가로 크기 (이미지 너비의 절반)
            rect_height = H // 6  # 직사각형의 세로 크기 (이미지 높이의 1/4)
        if(T<10):
            rect_width = W // 3  # 직사각형의 가로 크기 (이미지 너비의 절반)
            rect_height = H // 5  # 직사각형의 세로 크기 (이미지 높이의 1/4)
        if(T<15):
            rect_width = W // 2  # 직사각형의 가로 크기 (이미지 너비의 절반)
            rect_height = H // 4  # 직사각형의 세로 크기 (이미지 높이의 1/4)


        # 좌표 설정
        # 직사각형 그리기
        if(T <5):
            x0 = cx - rect_width // 2
            x1 = cx + rect_width // 2
            y0 = cy - rect_height // 2
            y1 = cy + rect_height // 2
        elif(T<10):
            x0 = cx - rect_width // 3
            x1 = cx + rect_width // 3
            y0 = cy - rect_height // 3
            y1 = cy + rect_height // 3
        else:
            x0 = cx - rect_width // 4
            x1 = cx + rect_width // 4
            y0 = cy - rect_height // 4
            y1 = cy + rect_height // 4

        # 마스크에 직사각형 그리기
        mask[:, y0:y1, x0:x1, :] = 1

        # 마스크의 형태 수정
        return mask

    # 입력 이미지의 크기를 가져옵니다.
    height = FLAGS.img_shapes[0]  # 이미지의 높이
    width = FLAGS.img_shapes[1]  # 이미지의 너비

    # 마스크 생성
    mask = generate_mask(height, width,test_num)
    test_num +=1
    return mask  # 마스크 반환
