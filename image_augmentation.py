import os
import shutil
import PIL
from PIL import Image
import numpy as np
import time

ROOT_PATH = 'C:/Users/joonh/Desktop/project'

IMG_PATH = 'images'

img_path = f'{ROOT_PATH}/{IMG_PATH}'
img_list = os.listdir(img_path)

if not os.path.exists(f'{ROOT_PATH}/img_aug') :
    os.mkdir(f'{ROOT_PATH}/img_aug')

#print(img_list)
for img_file in img_list :
    print(img_file)
    image = Image.open(f'{img_path}/{img_file}')
    try :
        # RGB
        # if image.mode in ("RGBA"):
        #     iamge = image.convert("RGB")
        # img_name = img_file.split('.jpg')[0]
        # print(img_name)

        # r, g, b = image.split()
        # r.save(f'{ROOT_PATH}/img_crop/{img_name}_r.jpg')
        # g.save(f'{ROOT_PATH}/img_crop/{img_name}_g.jpg')
        # b.save(f'{ROOT_PATH}/img_crop/{img_name}_b.jpg')
        
        # #회전
        # resize_image = image.resize((480,480))
        # if not os.path.exists(f'{ROOT_PATH}/img_aug/{img_name}_480.jpg') :
        #         resize_image.save(f'{ROOT_PATH}/img_aug/{img_name}_480.jpg')
        # for r in range(0, 360, 90) : 
        #     if r == 180: 
        #         continue
        #     rotated_image = image.rotate(r)
        #     # rotated_image_array = np.array(rotated_image)
        #     # rotated_image = Image.fromarray(rotated_image_array)
        #     if not os.path.exists(f'{ROOT_PATH}/img_aug/{img_name}_{r}.jpg') :
        #         rotated_image.save(f'{ROOT_PATH}/img_aug/{img_name}_{r}.jpg')

        # wid = image.width
        # hig = image.height

        # print(wid)
        # print(hig)

        # wid_cut = wid // 10 * 2
        # hig_cut = hig // 10 * 2

        # x1, y1, x2,y2 = wid_cut *1, hig_cut * 1, wid_cut * 3, hig_cut * 3
        # x1, y1, x2,y2 = wid_cut *2, hig_cut * 1, wid_cut * 4, hig_cut * 3
        # x1, y1, x2,y2 = wid_cut *1, hig_cut * 2, wid_cut * 3, hig_cut * 4   
        # x1, y1, x2,y2 = wid_cut *2, hig_cut * 2, wid_cut * 4, hig_cut * 4


        crop_image = image.crop((x1,y1,x2,y2))
        crop_image.save(f'{ROOT_PATH}/img_crop/{img_name}_{1+i}{1+j}{3+i}{3+j}.jpg') 
        # time.sleep(100)

    except Exception as e:
         print(e)
         print("실패")
         time.sleep(0.5)
         pass
    

        
    #색 바꾸기

    #crop

    #