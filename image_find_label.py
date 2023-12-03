import os
import sys
import shutil

MOD='validation' # train or validation
ROOT_PATH='.'
IMAGE_PATH='img_aug/image'
LABEL_PATH='img_aug/label'
ORIGIN_DATA_PATH=f'origin_dataset/{MOD}'

# img_list=os.listdir(IMAGE_PATH)
# print(img_list)

path1=f'./origin_dataset/{MOD}'

car_list=os.listdir(path1)

# 기존 트림 블랙 어쩌구 밖으로 꺼내는거
def remove_long_trim():
    for car in car_list:
        img_path=f"{path1}/{car}/image"
        label_path=f"{path1}/{car}/label"

        list2=os.listdir(img_path)
        for i in list2:
            img_path2=f"{img_path}/{i}"
            try:  os.listdir(img_path2)
            except: continue

            for j in os.listdir(img_path2):
                try: shutil.copy(f"{img_path2}/{j}", f"{img_path}/{j}")
                except: pass

        list2=os.listdir(label_path)
        for i in list2:
            label_path2=f"{label_path}/{i}"
            try: os.listdir(label_path2)
            except: continue

            for j in os.listdir(label_path2):
                try: shutil.copy(f"{label_path2}/{j}", f"{label_path}/{j}")
                except: pass

# def get_label_files():


# def parse_image_files(feautre):

# remove_long_trim()

if __name__ == "__main__":
    car_list = os.listdir(ORIGIN_DATA_PATH)
    print(car_list)

    for car in car_list:
        label_src_path = f"{ORIGIN_DATA_PATH}/{car}/label"
        label_list = os.listdir(label_src_path)

        img_src_path = f"{IMAGE_PATH}/{car}/image"
        image_list=os.listdir(img_src_path)
        print(image_list)
        image_list = set(map(lambda file: file.split('.')[0], image_list))
        
        print(car)
        for label_name in label_list:
            label_name = label_name.split('.')[0]
            if label_name in image_list:
                try: shutil.copy(f"{label_src_path}/{label_name}.json", f"{LABEL_PATH}/{label_name}.json")
                except Exception as e: print(e)
        