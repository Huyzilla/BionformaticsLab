import cv2
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from collections import deque
import math


# -------------------------------------------------------------
def Show_Histogram_Board(img):
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(max(histg[100:]))
    plt.plot(histg)
    plt.show(block=False)


def Find_suitable_threshold_ver2(img):
    h, bins = np.histogram(img.astype(np.uint8), range(255))
    h[:30] = [0] * 30
    L = 0
    R = 255
    M = (L + R) // 2
    BigSum = sum(h)
    while L < M < R:
        SmallSum = sum(h[:M])
        if SmallSum > BigSum * 0.7:
            R = M
        else:
            L = M
        M = (L + R) // 2
    return M


def Erode(img, size=5, repeat=1):
    kernel = np.ones((size, size), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(img, kernel, iterations=repeat)
    return img_erosion


def Dilitation(img, size=5, repeat=1):
    kernel = np.ones((size, size), np.uint8)
    img_dilitation = cv2.dilate(img, kernel, iterations=repeat)
    return img_dilitation


def Contour(img, show_process=False, origin_img=None):
    rows, cols = img.shape
    Min_y = 0
    Max_y = rows
    while not any(img[Min_y]):
        Min_y += 1
    while not any(img[Max_y - 1]):
        Max_y -= 1
    Min_x = 0
    Max_x = cols
    while not any(img[Min_y:Max_y, Min_x]):
        Min_x += 1
    while not any(img[Min_y:Max_y, Max_x - 1]):
        Max_x -= 1
    if show_process:
        cv2.rectangle(origin_img, (Min_x, Min_y), (Max_x, Max_y), (255, 255, 0), 5)
    Big_S = (Max_y - Min_y) * (Max_x - Min_x)
    img = ~img
    contours, hierarchy = cv2.findContours(img, 1, 2)
    Big_box = [[None]]
    Areas = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        area = cv2.contourArea(box)
        Areas.append(area / Big_S)
        if Big_S * 0.2 <= area <= Big_S * 0.3:
            Big_box = box.copy()
            if show_process:
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

                cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.drawContours(origin_img, [box], 0, (255, 0, 0), 5)

                cv2.circle(origin_img, leftmost, 5, (0, 0, 255), -1)
                cv2.circle(origin_img, rightmost, 5, (0, 0, 255), -1)
                cv2.circle(origin_img, topmost, 5, (0, 0, 255), -1)
                cv2.circle(origin_img, bottommost, 5, (0, 0, 255), -1)

                hull = cv2.convexHull(cnt, returnPoints=False)
                defects = cv2.convexityDefects(cnt, hull)

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv2.line(img, start, end, [0, 255, 0], 2)
                    cv2.circle(img, far, 5, [0, 255, 255], -1)
            break
    if show_process:
        plt.figure(figsize=(10, 10))
        plt.imshow(origin_img, cmap=plt.cm.gray)
        plt.show()
    return Big_box


def Binary_Threshold(img, threshold=180):
    ret, th2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return th2


def Load_img(win_path):
    path_universal = Path(win_path)
    img = cv2.imread(path_universal, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img


def Load_img_in_color(win_path):
    path_universal = Path(win_path)
    img = cv2.imread(path_universal)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img


def Count_nonzero_num(Big_box, temp_bin_img, show_process=False):
    # Root_x, Root_y = Big_box[0]
    # Vector_X = (Big_box[1][0] - Root_x, Big_box[1][1] - Root_y)
    # Vector_Y = (Big_box[3][0] - Root_x, Big_box[3][1] - Root_y)

    # Small_boxes = list()

    # Offset = [0, 9 / 38, 0.5, 29 / 38, 1]
    # for offset_x in range(1, 5):
    #     for offset_y in range(1, 5):
    #         Temp_arr = list()
    #         for i, j in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
    #             Temp_arr.append(
    #                 [
    #                     round(
    #                         Root_x
    #                         + Vector_X[0] * Offset[offset_x + i]
    #                         + Vector_Y[0] * Offset[offset_y + j]
    #                     ),
    #                     round(
    #                         Root_y
    #                         + Vector_X[1] * Offset[offset_x + i]
    #                         + Vector_Y[1] * Offset[offset_y + j]
    #                     ),
    #                 ]
    #             )

    #         Small_boxes.append(np.array(Temp_arr))

    # ------------- NEW WAY TO CAL SMALL BOXES
    Offset = [0, 9 / 38, 0.5, 29 / 38, 1]
    Small_boxes = list()
    Vector_AB = [Big_box[1][0] - Big_box[0][0], Big_box[1][1] - Big_box[0][1]]
    Vector_DC = [Big_box[2][0] - Big_box[3][0], Big_box[2][1] - Big_box[3][1]]
    Main_Lines = list()
    for i in Offset:
        A_Point = [Big_box[0][0] + Vector_AB[0] * i, Big_box[0][1] + Vector_AB[1] * i]
        C_Point = [Big_box[3][0] + Vector_DC[0] * i, Big_box[3][1] + Vector_DC[1] * i]
        Main_Lines.append((A_Point, C_Point[0] - A_Point[0], C_Point[1] - A_Point[1]))
    for OffsetID in range(1, 5):
        for MainLineID in range(1, 5):
            Temp_arr = list()
            for i, j in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
                Temp_arr.append(
                    [
                        round(
                            Main_Lines[MainLineID + i][0][0]
                            + Main_Lines[MainLineID + i][1] * Offset[OffsetID + j]
                        ),
                        round(
                            Main_Lines[MainLineID + i][0][1]
                            + Main_Lines[MainLineID + i][2] * Offset[OffsetID + j]
                        ),
                    ]
                )
            Small_boxes.append(np.array(Temp_arr))
    # print(Small_boxes)
    for box in Small_boxes:
        cv2.drawContours(temp_bin_img, [box], 0, 0, 4)
    if show_process:
        print("Adjust img--------------")
        plt.figure(figsize=(10, 10))
        plt.imshow(temp_bin_img, cmap=plt.cm.gray)
        plt.show()
    return np.count_nonzero(temp_bin_img)


def Adjust_Big_square_coordinates(Big_box, bin_img, show_process=False):
    Stop = False
    Loop_time = 0
    Centre_of_Big_box = [
        sum([Big_box[i][0] for i in range(4)]) / 4,
        sum([Big_box[i][1] for i in range(4)]) / 4,
    ]
    Important_area = Big_box.copy()
    for i in range(4):
        Important_area[i][0] = (
            Centre_of_Big_box[0] + (Big_box[i][0] - Centre_of_Big_box[0]) * 0.95
        )
        Important_area[i][1] = (
            Centre_of_Big_box[1] + (Big_box[i][1] - Centre_of_Big_box[1]) * 0.95
        )

    maskImage = np.zeros(bin_img.shape, dtype=np.uint8)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(maskImage, cmap=plt.cm.gray)
    # plt.show()
    cv2.drawContours(maskImage, [Important_area], 0, 255, -1)
    bin_img = cv2.bitwise_and(bin_img, maskImage)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(bin_img, cmap=plt.cm.gray)
    # plt.show()

    temp_bin_img = bin_img.copy()
    NonZero_num = Count_nonzero_num(Big_box, temp_bin_img)
    if show_process:
        print("====== Non Zero num: ", NonZero_num, "    =============== !")

    while not Stop:
        Best_way = [-1, -1, -1]
        for i in range(4):
            for offset_x in range(-10, 11, 1):
                for offset_y in range(-10, 11, 1):
                    if offset_x == offset_y == 0:
                        continue
                    temp_bin_img = bin_img.copy()
                    Big_box[i][0] += offset_x
                    Big_box[i][1] += offset_y
                    temp_NonZero_num = Count_nonzero_num(
                        Big_box, temp_bin_img, show_process=False
                    )
                    if temp_NonZero_num < NonZero_num:
                        NonZero_num = temp_NonZero_num
                        Best_way = [offset_x, offset_y, i]
                        # Count_nonzero_num(Big_box, temp_bin_img, show_process=True)
                    Big_box[i][0] -= offset_x
                    Big_box[i][1] -= offset_y
        if Best_way != [-1, -1, -1]:

            Big_box[Best_way[2]][0] += Best_way[0]
            Big_box[Best_way[2]][1] += Best_way[1]
            if show_process:
                print("Best way: ", Best_way)
                Count_nonzero_num(Big_box, temp_bin_img, show_process=True)
        else:
            break
        Loop_time += 1
        if Loop_time == 50:
            Stop = True
            if show_process:
                print("----    Stop by time out    ------------------")
    if show_process:
        print("Best Ans:", NonZero_num, "   =============\n\n")
    return Big_box


def Process_with_path(win_path, show_process=False):
    img = Load_img(win_path)
    origin_img = Load_img_in_color(win_path)
    rows, cols = img.shape
    if show_process:
        Show_Histogram_Board(img)
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()
    Threshold = Find_suitable_threshold_ver2(img)
    if show_process:
        print("------- Threshold: ", Threshold)
    img = Binary_Threshold(img, Threshold)
    if show_process:
        Show_Histogram_Board(img)
        print("---AFTER THRESHOLD----")
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()
    BigSquareImg = Dilitation(img, size=3, repeat=6)
    if show_process:
        plt.figure(figsize=(10, 10))
        plt.imshow(BigSquareImg, cmap=plt.cm.gray)
        plt.show()

    BigSquareImg = Erode(BigSquareImg, size=3, repeat=12)
    if show_process:
        plt.figure(figsize=(10, 10))
        plt.imshow(BigSquareImg, cmap=plt.cm.gray)
        plt.show()
    BigSquareImg = Dilitation(BigSquareImg, size=3, repeat=6)
    if show_process:
        print("FINAL---------------")
        plt.figure(figsize=(10, 10))
        plt.imshow(BigSquareImg, cmap=plt.cm.gray)
        plt.show()

    Big_box = Contour(
        BigSquareImg, show_process=show_process, origin_img=origin_img.copy()
    )
    Big_box = Adjust_Big_square_coordinates(Big_box, img, show_process=show_process)

    # Root_x, Root_y = Big_box[0]
    # Vector_X = (Big_box[1][0] - Root_x, Big_box[1][1] - Root_y)
    # Vector_Y = (Big_box[3][0] - Root_x, Big_box[3][1] - Root_y)
    # Small_boxes = list()
    # Offset = [0, 9 / 38, 0.5, 29 / 38, 1]
    # for offset_x in range(1, 5):
    #     for offset_y in range(1, 5):
    #         Temp_arr = list()
    #         for i, j in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
    #             Temp_arr.append(
    #                 [
    #                     round(
    #                         Root_x
    #                         + Vector_X[0] * Offset[offset_x + i]
    #                         + Vector_Y[0] * Offset[offset_y + j]
    #                     ),
    #                     round(
    #                         Root_y
    #                         + Vector_X[1] * Offset[offset_x + i]
    #                         + Vector_Y[1] * Offset[offset_y + j]
    #                     ),
    #                 ]
    #             )
    #         Small_boxes.append(np.array(Temp_arr))

    # ------------- NEW WAY TO CAL SMALL BOXES
    Offset = [0, 9 / 38, 0.5, 29 / 38, 1]
    Small_boxes = list()
    Vector_AB = [Big_box[1][0] - Big_box[0][0], Big_box[1][1] - Big_box[0][1]]
    Vector_DC = [Big_box[2][0] - Big_box[3][0], Big_box[2][1] - Big_box[3][1]]
    Main_Lines = list()
    for i in Offset:
        A_Point = [Big_box[0][0] + Vector_AB[0] * i, Big_box[0][1] + Vector_AB[1] * i]
        C_Point = [Big_box[3][0] + Vector_DC[0] * i, Big_box[3][1] + Vector_DC[1] * i]
        Main_Lines.append((A_Point, C_Point[0] - A_Point[0], C_Point[1] - A_Point[1]))
    for OffsetID in range(1, 5):
        for MainLineID in range(1, 5):
            Temp_arr = list()
            for i, j in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
                Temp_arr.append(
                    [
                        round(
                            Main_Lines[MainLineID + i][0][0]
                            + Main_Lines[MainLineID + i][1] * Offset[OffsetID + j]
                        ),
                        round(
                            Main_Lines[MainLineID + i][0][1]
                            + Main_Lines[MainLineID + i][2] * Offset[OffsetID + j]
                        ),
                    ]
                )
            Small_boxes.append(np.array(Temp_arr))
    # print(Small_boxes, "<=======")
    if show_process:
        color = [(255, 0, 0)] + [(0, 0, 255), (0, 0, 255), (0, 0, 255)] * 6
        i = 0
        for box in Small_boxes:
            cv2.drawContours(origin_img, [box], 0, color[i], 2)
            i += 1
        plt.figure(figsize=(10, 10))
        plt.imshow(origin_img, cmap=plt.cm.gray)
        plt.show()
    if show_process:
        return origin_img
    return Big_box, Small_boxes


def isInsideTriangle(A, B, C, P):
    # print("A:", A)
    # print("P:", P)
    # Calculate the barycentric coordinates
    # of point P with respect to triangle ABC
    denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    a = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denominator
    b = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denominator
    c = 1 - a - b

    # Check if all barycentric coordinates
    # are non-negative
    if a >= 0 and b >= 0 and c >= 0:
        return True
    else:
        return False


def Check_inside(Point, Box):
    return any(
        [
            isInsideTriangle(Box[0], Box[1], Box[2], Point),
            isInsideTriangle(Box[0], Box[2], Box[3], Point),
        ]
    )


def Cal_dist(A, B):
    return math.sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2))


def Check_in_line(A, B, C, threshold):
    if (Cal_dist(C, A) + Cal_dist(C, B)) / Cal_dist(A, B) < threshold:
        return True
    return False


def In_which_box(contours, Small_Boxes):
    Vote = [0] * 17
    for Point in contours:
        # print("Point: ", Point)
        for i in range(16):
            if Check_inside(Point[0], Small_Boxes[i]):
                Vote[i] += 1
                if i in (0, 1, 2, 3):
                    if Vote[i] > 11:
                        if Check_in_line(
                            Small_Boxes[i][0], Small_Boxes[i][1], Point[0], 1.1
                        ):
                            Vote[i] -= 5
                if i in (0, 4, 8, 12):
                    if Vote[i] > 11:
                        if Check_in_line(
                            Small_Boxes[i][3], Small_Boxes[i][0], Point[0], 1.1
                        ):
                            Vote[i] -= 5
                if i in (3, 7, 11, 15):
                    if Check_in_line(
                        Small_Boxes[i][1], Small_Boxes[i][2], Point[0], 1.1
                    ):
                        Vote[i] += 11
                if i in (12, 13, 14, 15):
                    if Check_in_line(
                        Small_Boxes[i][2], Small_Boxes[i][3], Point[0], 1.1
                    ):
                        Vote[i] += 11
                break
        else:
            Vote[16] += 1
    Max_vote = max(Vote)
    if Max_vote > 0:
        Index = Vote.index(Max_vote)
        if Index == 16:
            return None
        else:
            return Vote.index(Max_vote)
    return None


def Show_Contour(Origin_path, Mask_path):
    Mask_img_gray = Load_img(Mask_path)
    Origin_img = Load_img_in_color(Origin_path)
    contours, hierarchy = cv2.findContours(
        Mask_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        cv2.drawContours(Origin_img, [c], 0, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(Origin_img, cmap=plt.cm.gray)
    plt.show()


def Count_Yeast_in_16_Squares(Origin_path, Mask_path, show_process=False):
    Big_box, Small_boxes = Process_with_path(Origin_path)
    # Mask_img = Load_img_in_color(Mask_path)
    Mask_img_gray = Load_img(Mask_path)
    Origin_img = Load_img_in_color(Origin_path)
    # rows, cols = Mask_img_gray.shape
    # Big_S = rows * cols
    Big_S = cv2.contourArea(Big_box)
    Min_S_limit = Big_S * 0.00015
    Max_S_limit = Big_S * 0.01
    if show_process:
        # print("Max/Min S limit", Max_S_limit, Min_S_limit)
        for box in Small_boxes:
            cv2.drawContours(Origin_img, [box], 0, (94, 53, 177), 2)
    contours, hierarchy = cv2.findContours(
        Mask_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # print(sorted([cv2.contourArea(c) for c in contours]))
    Colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)] * 6
    Count_yeast = [0] * 16
    for c in contours:
        # print(c)
        # calculate moments for each contour
        area = cv2.contourArea(c)
        # print(area)
        if area < Min_S_limit or area > Max_S_limit:
            continue
        # print(area)
        M = cv2.moments(c)
        # if M["m00"] == 0:
        #     continue
        In_box = In_which_box(c, Small_boxes)
        if In_box != None:
            Count_yeast[In_box] += 1
            # calculate x,y coordinate of center
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            # cv2.circle(Origin_img, (cX, cY), 3, Colors[In_box], -1)
            cv2.drawContours(Origin_img, [c], 0, Colors[In_box], 2)
    if show_process:
        Color_for_direction = [
            (255, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 255, 0),
        ]
        Direction = [(3, 0), (0, 1), (1, 2), (2, 3)]
        for i in range(4):
            cv2.line(
                Origin_img,
                Small_boxes[0][Direction[i][0]],
                Small_boxes[0][Direction[i][1]],
                Color_for_direction[i],
                2,
            )
        Colors = [(0, 0, 125), (125, 0, 0), (0, 125, 0)] * 6
        for index in range(16):
            cv2.putText(
                Origin_img,
                "{}-{}".format(index + 1, Count_yeast[index]),
                (
                    sum(i[0] for i in Small_boxes[index]) // 4 - 25,
                    sum(i[1] for i in Small_boxes[index]) // 4,
                ),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                1,
                Colors[index],
                4,
            )
        squares_info = []
        for i, box in enumerate(Small_boxes):
            coords = [{"x": int(p[0]), "y": int(p[1])} for p in box]
            squares_info.append({
                "id": f"cell_{i+1}",
                "coordinates": coords,
                "yeast_count": Count_yeast[i]
            })
        # plt.figure(figsize=(10, 10))
        # plt.imshow(Origin_img, cmap=plt.cm.gray)
        # plt.show()
        return Origin_img, squares_info
    else:
        return Count_yeast
