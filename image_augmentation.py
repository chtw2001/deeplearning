import os
import shutil
import PIL
from PIL import Image
import numpy as np
import time
import cv2
import json
import random

from multiprocessing import Pool

random.seed(10)

# 디렉토리 구조 : # deeplearning/crop_dataset/[차종들]/[이미지들]
ROOT_PATH = 'C:/Users/joonh/Desktop/mlproject/deeplearning/img_aug' # yolo2.py 있는 메인 디렉토리

IMG_PATH = 'image' # 그안에 crop_dataset 디렉토리
LABEL_PATH = 'label'
RESULT_PATH='result_gray_and_weather_train'

id_tag = {
    "P00": "car_all",
    "P01": "front_bumper",
    "P02": "rear_bumper",
    "P03": "tire",
    "P04": "A_filler",
    "P05": "C_filler",
    "P06": "side_mirror",
    "P07": "front_door",
    "P08": "back_door",
    "P09": "radiator_grill",
    "P10": "head_lamp",
    "P11": "rear_lamp",
    "P12": "bonnet",
    "P13": "trunk",
    "P14": "roop"
}
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


def get_weather_image(params):
    car_image, save_path, image_name = params
    rain_dir = "C:/Users/joonh/Desktop/mlproject/deeplearning/img_aug/rain"
    rain_files = os.listdir(rain_dir)
    for i in range(2):
        # input_path = os.path.join(input_dir, input_file)

        # 랜덤하게 하나의 빗방울 이미지 파일 선택
        rain_file = random.choice(rain_files)
        rain_path = os.path.join(rain_dir, rain_file)

        # 빗방울 이미지 열기
        rain_image = Image.open(rain_path)

        # 원본 차 이미지 크기 가져오기
        car_width, car_height = car_image.size

        # 빗방울 이미지의 크기와 차 이미지의 크기를 비교하여 크롭 여부 결정
        # if car_width <= rain_image.width and car_height <= rain_image.height:
            # 빗방울 이미지를 크롭할 범위 계산
        rain_image=rain_image.resize((car_width * 2, car_height * 2))
        crop_x_range = rain_image.width - car_width
        crop_y_range = rain_image.height - car_height

        # 크롭 범위가 음수일 경우, 0으로 설정
        crop_x = max(0, random.randint(0, crop_x_range))
        crop_y = max(0, random.randint(0, crop_y_range))

        # 빗방울 이미지 크롭
        cropped_rain_image = rain_image.crop((crop_x, crop_y, crop_x + car_width, crop_y + car_height))

        # 빗방울 이미지의 투명도 조절
        alpha = random.uniform(0.1, 0.5)  # 원하는 투명도 값 (0.0 ~ 1.0)
        # alpha=0.3
        cropped_rain_image = cropped_rain_image.convert('RGBA')
        for x in range(cropped_rain_image.width):
            for y in range(cropped_rain_image.height):
                r, g, b, a = cropped_rain_image.getpixel((x, y))
                cropped_rain_image.putpixel((x, y), (r, g, b, int(a * alpha)))

        # 빗방울 이미지를 원본 차 이미지 위에 합성
        combined_image = Image.alpha_composite(car_image.convert('RGBA'), cropped_rain_image)
        combined_image=combined_image.convert('RGB')
        # 출력 디렉토리에 저장
        output_path = f"{save_path}/{image_name}_weather_{i}.jpg"
        combined_image.save(output_path, 'JPEG')  # 이미지를 PNG 형식으로 저장 (EPS로 저장하려면 다른 방식 필요)
        print(f"[Success] : Save image - {output_path}")



def change_color_filters():
    class_path = f'{ROOT_PATH}/{IMG_PATH}'
    image_class = os.listdir(class_path)

    for class_name in image_class:
        print(class_name)

        image_name_list = os.listdir(f"{class_path}/{class_name}")
        
        car_images = []
        for image_name in image_name_list:
            save_path = f"{ROOT_PATH}/{RESULT_PATH}/{class_name}"

            # print(save_path)
            if not os.path.exists(save_path) :
                os.makedirs(save_path)

            image_path = f"{class_path}/{class_name}/{image_name}"
            print(image_path)

            if os.path.isdir(image_path):
                continue

            
            # # PIL ----------------------------------------------
            try:
                car_image = Image.open(image_path).convert("L")
                car_images.append((car_image, save_path, image_name))
                # get_weather_image(car_image, save_path, image_name)

            except Exception as e:
                print(e)
                pass
            
                # 원본 차 이미지 열기
                

            # CV2 ------------------------------------------------------------
            # try:
                # 뭔가 테두리만 추출하는거? 써먹을 수 있을진 모르겠음
            cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # cv2.imwrite(f"{save_path}/{image_name}_gray.jpg", cv_image)

            # # 엣지 머시기
            # threshold1 = 0
            # threshold2 = 360
            # edge_img = cv2.Canny(cv_image, threshold1, threshold2)
            # cv2.imwrite('./test.jpg', edge_img)
            
            # 뿌옇게 만들기
            gaussian_img = cv2.GaussianBlur(cv_image, (35,35), 0)
            cv2.imwrite(f"{save_path}/{image_name}_gaussian.jpg", gaussian_img)
            
            # 이미지 뭉게기?
            median_img = cv2.medianBlur(cv_image, 11)
            cv2.imwrite(f"{save_path}/{image_name}_median.jpg", median_img)
                

            # except Exception as e:
            #     print(e)
            #     pass
            # pass
        multi_pool = Pool(12)
        multi_pool.map(get_weather_image, car_images)


def crop_feature():
    car_list = os.listdir(f"{ROOT_PATH}/{IMG_PATH}")
    for car in car_list:
        img_files = os.listdir(f"{ROOT_PATH}/{IMG_PATH}/{car}")
        for img_file_name in img_files:
            img_file_path=f"{ROOT_PATH}/{IMG_PATH}/{car}/{img_file_name}"
            label_file_path=f"{ROOT_PATH}/{LABEL_PATH}/{img_file_name.split('.')[0]}.json"
            # print(img_file_path)
            # print(label_file_path)
            
            with open(label_file_path, 'r', encoding='utf-8') as f:
                label_config=json.load(f)
                label_config=label_config['learningDataInfo']['objects']

            # for class_ids
            # from pprint import pprint
            # pprint(label_config)
            # print(label_config)

            for car_part_class in label_config:
                id=car_part_class['classId']
                part_point=car_part_class['coords']
                x1, y1 = part_point['tl']['x'], part_point['tl']['y']
                x2, y2 = part_point['br']['x'], part_point['br']['y']

                cv_image = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
                crop_image = cv_image[int(y1):int(y2), int(x1):int(x2)]

                # cv2.imshow('crop', crop_image)
                # cv2.waitKey(0)
                
                id_result_path=f"{ROOT_PATH}/{RESULT_PATH}/{id_tag[id.split('.')[0]]}/{car}"

                try:
                    os.makedirs(id_result_path)
                except:
                    pass

                status = cv2.imwrite(f"{id_result_path}/{img_file_name.split('.')[0]}_crop.jpg", crop_image)
                
                print(f"{id_result_path}/{img_file_name.split('.')[0]}_crop.jpg : {status}")
                # time.sleep(1)
                # pil_image = Image.open(img_file_path)
                # crop_image = crop(pil_image, x1, y1, x2, y2)
                # crop_image.show()

                
def split_train():
    root_path = 'C:\\Users\\joonh\\Desktop\\mlproject\\deeplearning\\img_aug\\result_gray_and_weather_train'
    car_class = os.listdir(root_path)
    print(car_class)
    for car in car_class:
        source_directory = root_path + f'\\{car}'
        target_directory_train = root_path + f'\\train\\{car}'
        target_directory_val = root_path + f'\\validation\\{car}'
        
        try : os.makedirs(target_directory_train)
        except : pass
        try : os.makedirs(target_directory_val)
        except : pass
        split_size=0.7
        files = os.listdir(source_directory)
        print(files[:5])

        # 파일 리스트를 섞는다.
        random.shuffle(files)
        print(files[:5])

        # 학습 데이터와 검증 데이터의 분할 지점을 설정한다.
        split_point = int(len(files) * split_size)
        
        # 파일들을 학습 데이터와 검증 데이터로 분할한다.
        train_files = files[:split_point]
        val_files = files[split_point:]
        # 학습 데이터를 새 디렉토리에 복사한다.
        for file in train_files:
            shutil.copy(os.path.join(source_directory, file), target_directory_train)
        
        # 검증 데이터를 새 디렉토리에 복사한다.
        for file in val_files:
            shutil.copy(os.path.join(source_directory, file), target_directory_val)

def pie_chart():
    import matplotlib.pyplot as plt
    root_path = 'C:\\Users\\joonh\Desktop\\mlproject\\deeplearning\\training_dataset\\train'
    car_class = os.listdir(root_path)
    class_cnt = []
    for car in car_class:
        class_cnt.append(len(os.listdir(root_path + f'\\{car}')))

    plt.title("Class Percent")
    plt.pie(class_cnt, labels=car_class, autopct='%.1f%%', colors=plt.cm.Pastel1.colors)
    plt.show()

if __name__ == "__main__":
    # change_color_filters()
    # crop_feature()            
    # split_train()
    pie_chart()

    
