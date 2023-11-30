import os
import numpy as np
from PIL import Image
import json

# 이미지 폴더와 라벨 폴더 경로 설정
image_folder_path = 'C:/Users/nam02/Desktop/대학/3학년 2학기/머신러닝/프로젝트/deeplearning/crop_dataset/car_image'
label_folder_path = 'C:/Users/nam02/Desktop/대학/3학년 2학기/머신러닝/프로젝트/deeplearning/crop_dataset/traning_croped_car_label'

# 이미지 파일 목록 가져오기
image_files = os.listdir(image_folder_path)

# 이미지와 라벨 데이터를 저장할 리스트 초기화
images = []
labels = []

# 각 이미지 파일에 대한 라벨 파일을 찾아서 데이터를 불러오기
for image_file in image_files:
    # 이미지 파일 경로
    image_path = os.path.join(image_folder_path, image_file)

    # 라벨 파일 경로 (이미지 확장자를 제외하고 .txt 확장자로 변경)
    label_file = os.path.splitext(image_file)[0] + '.json'
    label_path = os.path.join(label_folder_path, label_file)

    # 이미지 로드
    image = Image.open(image_path)
    # 이미지를 모델 입력 크기로 조정 (예: 224x224)
    image = image.resize((224, 224))
    # 이미지를 NumPy 배열로 변환
    image_array = np.array(image)

    # 라벨 파일 로드 (라벨 파일에는 클래스 정보가 있어야 함)
    with open(label_path, 'r', encoding='utf-8') as label_file:
        label_data = json.load(label_file)



    # 라벨을 정수 또는 클래스 이름으로 파싱하여 저장
    # 예: label = 'sedan' 또는 label = '0' (라벨이 문자열 또는 정수로 표현됨)
    labels.append(label_data["rawDataInfo"]["SmallCategoryId"])

    # 이미지와 라벨 데이터 리스트에 추가
    images.append(image_array)
    

# 이미지와 라벨 데이터를 NumPy 배열로 변환
images = np.array(images)
labels = np.array(labels)

# 고유한 클래스 추출
unique_labels = np.unique(labels)

# 클래스 수
num_classes = len(unique_labels)

# 라벨을 원-핫 인코딩으로 변환
one_hot_encoded_labels = np.zeros((len(labels), num_classes), dtype=int)

for i, label in enumerate(labels):
    class_index = np.where(unique_labels == label)[0][0]
    one_hot_encoded_labels[i, class_index] = 1

# print(images, one_hot_encoded_labels)
np.save('crop_dataset/images.npy', images)
np.save('crop_dataset/one_hot_encoded_labels.npy', one_hot_encoded_labels)