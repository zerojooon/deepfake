from mtcnn import mtcnn
import numpy as np
from PIL import ImageEnhance, Image


def crop(image, face_box, MUL=1.5):
    (x, y, w, h) = face_box
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    nw = int(round(w * MUL))
    nh = int(round(h * MUL))
    x_p = int(x - round((MUL - 1) * w / 2))
    y_p = int(y - round((MUL - 1) * h / 2))
    # print("test coordinates")
    # print(x_p, y_p, x_p+nw, y_p+nh)

    if 0 <= x_p <= 1920 and 0 <= y_p <= 1080 and 0 <= x_p + nw <= 1920 and 0 <= y_p + nh <= 1080:
        result_face = image[y_p:y_p + nh, x_p: x_p + nw]
    else:
        result_face = image[y:y + h, x:x + w]
    # elif x_p < 0 and y_p < 0:  #left down
    #     result_face = image[0:y_p + nh, 0: x_p + nw]
    # elif x_p < 0 and y_p > 1080: #left up
    #     result_face = image[y_p :1080, 0: x_p + nw]
    # elif x_p + nw > 1920 and y_p < 0: #right down
    #     result_face = image[y_p:y_p + nh, x_p: 1920]
    #     # print("abnormal1")
    # elif x_p + nw > 1920 and y_p + nh > 1080:
    #     result_face = image[y_p:1080, x_p:1080]
    # else:
    #     print(y,h,x,w)
    #     result_face = image[y:y+h, x:x+w]
    return result_face


def increase_brightness(img):  # increase brightness of image
    BR = 1.8
    CT = 0.9

    en1 = ImageEnhance.Brightness(img)
    image = en1.enhance(BR)

    en2 = ImageEnhance.Contrast(image)
    image = en2.enhance(CT)

    array = np.asarray(image)

    return array


# face_box 는 이전 프레임의 좌표를 인수로 받는 거
def crop_face(detector, img, face_box, h_size, v_size, width, height, ALPHA=0.5, run_mode=0):
    mid_x = (width -v_size) / 2
    mid_y = (height - h_size) / 2
    img = Image.fromarray(img)

    image = increase_brightness(img)
    face_list = detector.detect_faces(image)

    face_loc = None  # face가 없거나 있어도 잘못 detect된 경우라면 none으로 리턴하게끔
    result_face = np.array(None)

    detect_acc = 0
    detect_failed = 0
    center_loc = 0
    face_moved = 0
    iir_alpha = 0.8

    if len(face_list) == 0:
        detect_failed = 1

        if face_box == (mid_x, mid_y, h_size, v_size) and run_mode == 0:  # if n_frames<10: run_mode=0
            # print("crop center position")
            center_loc = 1
        elif face_box == (mid_x, mid_y, h_size, v_size):
            center_loc = 1
            face_loc = face_box
            result_face = crop(image, face_box)
        else:
            # print("no face detected, initiating new method" )
            face_loc = face_box
            result_face = crop(image, face_box)

    else:
        face_max = max([face['confidence'] for face in face_list])
        for face in face_list:
            if face['confidence'] == face_max:
                if face_box == (mid_x, mid_y, h_size, v_size):  # 만약 첫 번째 frame이라서 이전에 저장한 location이 없다면
                    face_loc = face['box']
                    result_face = crop(image, face['box'])

                else:
                    xp, yp, wp, hp = face_box[0], face_box[1], face_box[2], face_box[3]  # new location

                    x2p = xp + wp
                    y2p = yp + hp

                    (x, y, w, h) = face['box']
                    conf = face['confidence']
                    detect_acc = conf
                    x2 = x + w
                    y2 = y + h

                    if abs(xp - x) < wp * ALPHA and abs(yp - y) < hp * ALPHA and abs(x2p - x2) < wp * ALPHA and abs(
                            y2p - y2) < hp * ALPHA:
                        x_iir = (1 - iir_alpha) * x + iir_alpha * xp
                        y_iir = (1 - iir_alpha) * y + iir_alpha * yp
                        x2_iir = (1 - iir_alpha) * x2 + iir_alpha * x2p
                        y2_iir = (1 - iir_alpha) * y2 + iir_alpha * y2p

                        if (xp != x_iir):
                            if (x > xp):
                                x_iir = x_iir + 1
                            elif (x < xp):
                                x_iir = x_iir - 1

                        if (yp != y_iir):
                            if (y > yp):
                                y_iir = y_iir + 1
                            elif (y < yp):
                                y_iir = y_iir - 1

                        if (x2p != x2_iir):
                            if (x2 > x2p):
                                x2_iir = x2_iir + 1
                            elif (x2 < x2p):
                                x2_iir = x2_iir - 1

                        if (y2p != y2_iir):
                            if (y2 > y2p):
                                y2_iir = y2_iir + 1
                            elif (y2 < y2p):
                                y2_iir = y2_iir - 1

                        w_iir = x2_iir - x_iir
                        h_iir = y2_iir - y_iir

                        face['box'] = (x_iir, y_iir, w_iir, h_iir)
                        face_loc = face['box']
                        result_face = crop(image, face['box'])

                    else:
                        # print("It is not a face")
                        face_moved = 1
                        face_loc = face_box
                        result_face = crop(image, face_box)

    return face_loc, result_face, detect_acc, detect_failed, center_loc, face_moved