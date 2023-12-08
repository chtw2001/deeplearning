import easyocr

reader = easyocr.Reader(['ko'])  # 한국어 언어 설정

image_path = 'crop_dataset/ttest3.JPG'

results = reader.readtext(image_path)

# 각 텍스트 박스에서 텍스트 추출
for result in results:
    text, a, b = result
    print(a)
