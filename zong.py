import cv2
import numpy as np


def extract_leaves(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    

    b_channel = lab[:, :, 2]
    

    _, binary = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    black_background = np.zeros(binary.shape, dtype=np.uint8)
    

    outer_contour_idx = find_outer_contour_index(contours)
    if outer_contour_idx is not None:

        white_background = np.ones(black_background.shape, dtype=np.uint8) * 255

        cv2.drawContours(white_background, contours, outer_contour_idx, (0, 0, 0), thickness=cv2.FILLED)
        

        leaves_image = cv2.bitwise_not(white_background)
        return leaves_image
    
    return black_background


def find_outer_contour_index(contours):
    if len(contours) > 0:
        areas = [cv2.contourArea(contour) for contour in contours]
        outer_contour_idx = np.argmax(areas)
        return outer_contour_idx
    else:
        return None

def calculate_contour_areas(image_path):

    image = cv2.imread(image_path)


    leaves_image = extract_leaves(image)
    leaf_area = np.count_nonzero(leaves_image)  
    print("leaf area:", leaf_area)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


    a_channel = lab[:, :, 1]


    _, binary = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print("lunkuo{}area: {}".format(i+1, area))


    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 1000000:
            total_area += area


    coverage = (total_area / leaf_area) * 100
    print("coverage: {:.2f}%".format(coverage))


    return total_area


image_path = '------'
total_area = calculate_contour_areas(image_path)
print("area(all):", total_area)