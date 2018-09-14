import numpy as np
import keras
import cv2
import matplotlib.pyplot as plt

WEIGHTS_PATH = "/dog_weights.hdf5"
IMG_PATH = "E:/TensorFlow/image_classification/stanford_dogs/test/german_shepherd.png"
LABEL_PATH = "/labels.txt"

def InceptionResnetV2(num_classes=120):
    base_model = keras.applications.InceptionResNetV2(include_top=False, pooling='avg', weights=None)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
    model = keras.Model(base_model.inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def read_labels(path):
    text_file = open(path, "r")
    lines = text_file.read().split('\n')
    label_list = list()
    for line in lines:
        label_list.append(line)
    return label_list

def predict(img_path):
    labels = read_labels(LABEL_PATH)
    model = InceptionResnetV2()
    model.load_weights(WEIGHTS_PATH)
    im = cv2.imread(img_path)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=0)
    print(im.shape)
    outcome = model.predict(im)
    # breed = "Breed: " + labels[outcome.argmax()]
    # probability = "Probability: " + str(round(outcome.max() * 100, 2)) + "%"
    breeds = []
    probability = []
    for pred in outcome:
        top_indices = pred.argsort()[-5:][::-1]
        for i in top_indices:
            breeds.append(labels[i])
            probability.append(pred[i])
    # print("Breed: " + labels[outcome.argmax()])
    # print("Probability: " + str(round(outcome.max() * 100, 2)) + "%")
    return breeds, probability

def main():
    predict(IMG_PATH)

if __name__ == '__main__':
    main()
