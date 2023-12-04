import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 로드 및 전처리
images = np.load('crop_dataset/images.npy')
one_hot_encoded_labels = np.load('crop_dataset/one_hot_encoded_labels.npy')

# 데이터 전처리를 위해 평균과 표준 편차 계산
mean = np.mean(images, axis=(0, 1, 2))
std = np.std(images, axis=(0, 1, 2))

# 데이터 전처리 함수 정의
def preprocess_input(x):
    # 평균을 빼고 표준 편차로 나누어 스케일링 수행
    return (x - mean) / (std + 1e-7)

# ResNet-50 모델 불러오기
base_model = ResNet50(weights='imagenet', include_top=False)

# 모델 레이어 수정
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(
    preprocess_input(images),
    one_hot_encoded_labels,
    batch_size=32,
    epochs=10,
    validation_split=0.2  # 데이터를 학습 및 검증 세트로 나누세요.
)
model.save("my_model.h5")

# 모델 평가
test_datagen = ImageDataGenerator(
    # 테스트 데이터 전처리 옵션
)

test_generator = test_datagen.flow_from_directory(
    'path_to_test_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

evaluation = model.evaluate(test_generator)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

# 학습된 모델을 사용하여 예측 수행
new_images = ...  # 예측하고자 하는 이미지 데이터를 불러오세요.
preprocessed_new_images = preprocess_input(new_images)
predictions = model.predict(preprocessed_new_images)
