import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from seaborn import heatmap
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

modelPath = './model_saved/training_dataset/resnet_v50_50/'  # 모델이 저장된 경로
weight = 'model-044-0.592553-0.564767.h5'        # 학습된 모델의 파일이름
test_Path = './training_dataset/validation' # 테스트 이미지 폴더

def save_confusion(weight, test_Path):
    # modelPath = './model_saved/train/efficientnet_200/'  # 모델이 저장된 경로
    # modelPath = './model_saved/training_dataset/resnet_v50_10/'  # 모델이 저장된 경로
    # weight = 'model-010-0.372672-0.366883.h5'        # 학습된 모델의 파일이름
    # test_Path = './training_dataset/validation' # 테스트 이미지 폴더

    model = load_model(weight)
    datagen_test = ImageDataGenerator(rescale=1./255)
    generator_test = datagen_test.flow_from_directory(directory=test_Path,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    shuffle=False)

    # model로 test set 추론
    generator_test.reset()
    cls_test = generator_test.classes
    cls_pred = model.predict_generator(generator_test, verbose=1, workers=0)
    cls_pred_argmax = cls_pred.argmax(axis=1)

    # 결과 산출 및 저장
    report = metrics.classification_report(y_true=cls_test, y_pred=cls_pred_argmax, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(f'{modelPath}/report_test.csv', index=True, encoding='utf-8')
    print(report)

    report_con = metrics.confusion_matrix(y_true=cls_test, y_pred=cls_pred_argmax)
    heatmap(report_con, annot=True, fmt='d', linewidths=.5, cmap='YlGnBu')
    plt.xlabel("predict_class")
    plt.ylabel("class")
    plt.savefig(f'{modelPath}/confision_matrix.jpg', dpi=100)
    plt.show()


if __name__=="__main__":
    save_confusion(modelPath+weight, test_Path)