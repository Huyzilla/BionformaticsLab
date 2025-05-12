import cv2 as cv
import numpy as np


#Lấy mask chuẩn từ mask_using_unet
def get_watershed_mask(mask_using_unet):
    # Step 2: Convert the image to grayscale
    gray = mask_using_unet

    # Step 3: Apply thresholding to create a binary image
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thresh = cv.bitwise_not(thresh)

    # Step 4: Perform morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Step 5: Use the distance transform and thresholding to find sure foreground and background areas
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 6: Subtract the sure foreground from the sure background to get the unknown region
    unknown = cv.subtract(sure_bg, sure_fg)

    # Step 7: Perform connected components analysis to label the sure foreground
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    mask_using_unet = cv.cvtColor(mask_using_unet, cv.COLOR_GRAY2BGR)
    # Step 8: Apply the watershed algorithm to find the contours
    markers = cv.watershed(mask_using_unet, markers)

    # Step 9: Create a blank mask and fill the regions inside the contours with white color
    mask = np.zeros_like(gray)
    mask[markers > 1] = 255
    # cv2_imshow(mask)
    # Step 10: Find contours on the mask
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Step 11: Fill the contours on the original image with the same color
    # contour_img = mask.copy()
    for contour in contours:
        cv.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv.FILLED)
    return mask