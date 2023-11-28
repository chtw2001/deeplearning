import os
import shutil
import PIL
from PIL import Image
import numpy as np
import time
import torch
import cv2

res = torch.cuda.is_available()


# 디렉토리 구조 : # deeplearning/crop_dataset/[차종들]/[이미지들]
ROOT_PATH = 'C:/Users/joonh/Desktop/mlproject/deeplearning' # yolo2.py 있는 메인 디렉토리
 
IMG_PATH = 'crop_dataset' # 그안에 crop_dataset 디렉토리


# img_path = f'{ROOT_PATH}/{IMG_PATH}'
# img_list = os.listdir(img_path)

# if not os.path.exists(f'{ROOT_PATH}/img_aug') :
#     os.mkdir(f'{ROOT_PATH}/img_aug')


def crop(image, x1, y1, x2, y2):
    crop_image = image.crop((x1,y1,x2,y2))
    return crop_image


def convert_gray(image):
    image_gray = image.convert("L")
    return image_gray


def convert_black(image):
    image_gray = image.convert("1")
    return image_gray
    

def convert_RGB(image):
    if image.mode in ("RGBA"):
        image_rgb = image.convert("RGB")
        return image_rgb.split()
    else:
        return image
    

def resize(image, x_len, y_len):
    resize_image = image.resize((x_len,y_len))
    return resize_image


def rotate(image, angle, dir):
    if angle > 360:
        print("Angle 360 이하")
    if dir == 'left' or dir == 'l':
        rotate_image = image.rotate(angle)
    else:
        rotate_image = image.rotate(360 - angle)
    return rotate_image



if __name__ == "__main__":
    class_path = f'{ROOT_PATH}/{IMG_PATH}'
    image_class = os.listdir(class_path)

    for class_name in image_class:
        print(class_name)

        image_name_list = os.listdir(f"{class_path}/{class_name}")
        
        for image_name in image_name_list:
            save_path = f"{class_path}/{class_name}/augmentaion"

            if not os.path.exists(save_path) :
                os.makedirs(save_path)

            image_path = f"{class_path}/{class_name}/{image_name}"
            print(image_path)

            if os.path.isdir(image_path):
                continue

            # # PIL ----------------------------------------------
            # try:
            #     image = Image.open(image_path)

            #     # 회색 변환
            #     image_gray = convert_gray(image)
            #     # image_gray.show()
            #     image_gray.save(f"{save_path}/{image_name}_gray.jpg")

            #     # RGB 변환
            #     image_red, image_green, image_blue = convert_RGB(image)
            #     # image_red.show()
            #     image_red.save(f"{save_path}/{image_name}_red.jpg")
            #     # image_green.show()
            #     image_green.save(f"{save_path}/{image_name}_green.jpg")
            #     # image_blue.show()
            #     image_blue.save(f"{save_path}/{image_name}_blue.jpg")

            #     # 흑백? 비슷하게 변환
            #     image_black = convert_black(image)
            #     # image_black.show()
            #     image_black.save(f"{save_path}/{image_name}_black.jpg")

            #     # 회전, dir이 'left'거나 'l'이면 왼쪽으로 회전 아니면 우측회전
            #     angle = 90
            #     rotate_image = rotate(image,angle=angle, dir='r')
            #     rotate_image.show()
            #     rotate_image.save(f"{save_path}/{image_name}_{angle}.jpg")
            # except Exception as e:
            #     print(e)
            #     pass


            # CV2 ------------------------------------------------------------
            try:
                # 뭔가 테두리만 추출하는거? 써먹을 수 있을진 모르겠음
                cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(f"{save_path}/{image_name}_gray.jpg", cv_image)

                # 엣지 머시기
                # threshold1 = 0
                # threshold2 = 360
                # edge_img = cv2.Canny(cv_image, threshold1, threshold2)
                # cv2.imwrite('./test.jpg', edge_img)
                
                # 뿌옇게 만들기
                # gaussian_img = cv2.GaussianBlur(cv_image, (3,3), 0)
                # cv2.imwrite(f"{save_path}/{image_name}_gaussian.jpg", gaussian_img)
                
                # # 이미지 뭉게기?
                # median_img = cv2.medianBlur(cv_image, 11)
                # cv2.imwrite(f"{save_path}/{image_name}_median.jpg", median_img)
                

            except Exception as e:
                print(e)
                pass
            # pass
            
            

    
