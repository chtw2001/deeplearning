# 차량 분류 바운딩박스 작은거 분류
from PIL import Image
import cv2
from ultralytics import YOLO
SAVE_PATH = "/Users/chtw2001/Downloads/제네시스_G80/"
car_category = {'bicycle', 'motorcycle', 'bus', 'truck', 'car'}

def run_image_custom(img_path, individual_path):
    global SAVE_PATH
    global car_category
    try:
        model = YOLO('yolov8n.pt')

        results = model(img_path)
        bndboxs = results[0].boxes.data
        names = results[0].names

        # Find the index of the bounding box with the largest area
        largest_bbox_index = max(range(len(bndboxs)), key=lambda i: (bndboxs[i][2] - bndboxs[i][0]) * (bndboxs[i][3] - bndboxs[i][1]))

        # Extract coordinates of the largest bounding box
        largest_bbox = bndboxs[largest_bbox_index]
        xmin, ymin, xmax, ymax = map(int, largest_bbox[:4])

        # Check if the class is not a car, or if the bounding box is too small
        print(names[int(largest_bbox[5])])
        if names[int(largest_bbox[5])] not in car_category or (xmax - xmin) * (ymax - ymin) < 40000: # 박스 사이즈, 차량 분류 완료
            print("Not a car or bounding box too small. Skipping...")
            return

        img_array = results[0].orig_img.copy()
        cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('Car Detection', img_array)
        cv2.waitKey(0)
        cv2.imwrite("test.png", img_array)
        quit()

        image = Image.open(img_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        return

# Example usage
img_path = "./another.jpg"
individual_path = "path_to_individual_result.jpg"
run_image_custom(img_path, individual_path)