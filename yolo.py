# yolov8x
import cv2
from ultralytics import YOLO
import os
from PIL import Image
import time
# image_list = os.listdir(os.path.join("/Users/chtw2001/Downloads/제네시스G80_2021/2021_검정_트림A"))
SAVE_PATH = "/Users/chtw2001/Downloads/제네시스_G80/"


def run_imgage_custom(img_path, indivisual_path):
    global SAVE_PATH
    # Load YOLOv8 model
    try:
        # class 가 차가 아니면 pass, 박스 너무 작으면 pass
        model = YOLO('yolov8x.pt')

        # Run inference on the image
        results = model(img_path)
        # Access bounding box information
        bndboxs = results[0].boxes.data
        names = results[0].names

        # Find the index of the bounding box with the largest area
        largest_bbox_index = max(range(len(bndboxs)), key=lambda i: (bndboxs[i][2] - bndboxs[i][0]) * (bndboxs[i][3] - bndboxs[i][1]))

        # Extract coordinates of the largest bounding box
        largest_bbox = bndboxs[largest_bbox_index]
        xmin, ymin, xmax, ymax = map(int, largest_bbox[:4])

        # Draw the largest bounding box on the image
        img_array = results[0].orig_img.copy()
        cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Display and save the annotated image
        cv2.imshow('Car Detection', img_array)
        cv2.waitKey(0)
        cv2.imwrite("test.png", img_array)
        quit()
        # Print the coordinates of the largest bounding box
        # print(f"Largest Car Bounding Box Coordinates: {xmin}, {ymin}, {xmax}, {ymax}")  
        
        image = Image.open(img_path)
        
    except:
        return
    
    # try:
    #     crop_image = image.crop((xmin,ymin,xmax,ymax))
    #     crop_image.save(SAVE_PATH + indivisual_path) 
    #     print('image: ' + SAVE_PATH + indivisual_path)
    #     # time.sleep(100)

    # except Exception as e:
    #      print(e)
    #      print("실패")
    #      time.sleep(0.5)
    #      pass 
    
    
if __name__ == '__main__':
    big_path = "/Users/chtw2001/Downloads/car"
    midium_category = os.listdir(os.path.join(big_path))
    
    for small_category in midium_category:
        if small_category == '.DS_Store':
            continue
        
        midium_path = big_path + '/' + small_category
        car_list = os.listdir(os.path.join(midium_path))
        for image in car_list:
            if image == '.DS_Store':
                continue
            car_path = midium_path + '/' + image
            indivisual_list = os.listdir(os.path.join(car_path))
            for car in indivisual_list:
                if car == '.DS_Store':
                    continue
                indivisual_path = car_path + '/' + car
                run_imgage_custom(indivisual_path, car)
            quit()
            # a = input()
            # if a == 'y':
            #     continue
            # else:
            #     quit()



#-------------------
# # Run YOLOv8 inference on the image
# results = model(image)

# # Access the first element of the results list
# result = results[0]

# # Visualize the results on the image
# annotated_image = result.show()

# # Display the annotated image
# cv2.imshow("YOLOv8 Inference", annotated_image)

# # Wait for a key press and then close the display window
# cv2.waitKey(0)
# cv2.destroyAllWindows()





#---------------------
# # webcam 사용시
# cap = cv2.VideoCapture(0)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()


# def run_imgage_custom():
#     model = YOLO('yolov8n.pt')
#     results = model(img_path)
#     print(results)
    
#     bndboxs = results[0].boxes.data
#     class_id = results[0].boxes.cls
#     conf = results[0].boxes.conf
#     img_array = results[0].orig_img
#     names = results[0].names
#     print(f'names: {names}')
#     for i, bndbox in enumerate(bndboxs) :
#         xmin = int(bndbox[0])
#         ymin = int(bndbox[1])
#         xmax = int(bndbox[2])
#         ymax = int(bndbox[3])
#         conf = float(bndbox[4])
#         class_id = int(bndbox[5])
#         class_name = names[class_id]
#         print(xmin, ymin, xmax, ymax)
#         text = f"{class_name}-{round(conf, 2)}"
#         cv2. rectangle(img_array, (xmin, ymin), (xmax, ymax), (0,255,0), 2) 
#         cv2.putText(img_array, text, (xmin, ymin-5), cv2. FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2) 
        
#     cv2.imshow('car detection', img_array) 
#     cv2.waitKey(0) 
#     cv2.imwrite("test.png", img_array)
    