import os
import random
from PIL import Image

# 입력 디렉토리와 출력 디렉토리 설정
input_dir = 'crop_dataset/car_image/'
output_dir = 'crop_dataset/rain_car/'
rain_dir = 'crop_dataset/rain/'

# 입력 디렉토리에서 모든 파일 목록을 가져옴
input_files = os.listdir(input_dir)

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 빗방울 이미지 파일 목록 가져오기
rain_files = os.listdir(rain_dir)

for input_file in input_files:
    # 원본 차 이미지 열기
    input_path = os.path.join(input_dir, input_file)
    car_image = Image.open(input_path)

    # 랜덤하게 하나의 빗방울 이미지 파일 선택
    rain_file = random.choice(rain_files)
    rain_path = os.path.join(rain_dir, rain_file)

    # 빗방울 이미지 열기
    rain_image = Image.open(rain_path)

    # 원본 차 이미지 크기 가져오기
    car_width, car_height = car_image.size

    # 빗방울 이미지의 크기와 차 이미지의 크기를 비교하여 크롭 여부 결정
    if car_width <= rain_image.width and car_height <= rain_image.height:
        # 빗방울 이미지를 크롭할 범위 계산
        crop_x_range = rain_image.width - car_width
        crop_y_range = rain_image.height - car_height

        # 크롭 범위가 음수일 경우, 0으로 설정
        crop_x = max(0, random.randint(0, crop_x_range))
        crop_y = max(0, random.randint(0, crop_y_range))

        # 빗방울 이미지 크롭
        cropped_rain_image = rain_image.crop((crop_x, crop_y, crop_x + car_width, crop_y + car_height))

        # 빗방울 이미지의 투명도 조절
        alpha = 0.5  # 원하는 투명도 값 (0.0 ~ 1.0)
        cropped_rain_image = cropped_rain_image.convert('RGBA')
        for x in range(cropped_rain_image.width):
            for y in range(cropped_rain_image.height):
                r, g, b, a = cropped_rain_image.getpixel((x, y))
                cropped_rain_image.putpixel((x, y), (r, g, b, int(a * alpha)))

        # 빗방울 이미지를 원본 차 이미지 위에 합성
        combined_image = Image.alpha_composite(car_image.convert('RGBA'), cropped_rain_image)

        # 출력 디렉토리에 저장
        output_path = os.path.join(output_dir, input_file)
        combined_image.save(output_path, 'PNG')  # 이미지를 PNG 형식으로 저장 (EPS로 저장하려면 다른 방식 필요)
