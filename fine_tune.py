# Imports
import argparse
import math

import keras

SPLIT = 0.2

def InceptionResnetV2(num_classes):
    base_model = keras.applications.InceptionResNetV2(include_top=False, pooling='avg')
    outputs = keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
    model = keras.Model(base_model.inputs, outputs)
    for layer in model.layers[:500]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def get_args():
    # Get arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--classes", type=int)
    args = parser.parse_args()
    return args

def train(model, img_cols, img_rows, batch_size, data_dir, tensorboard, nb_epoch, model_path):
    # Create checkpoint to monitor training process and save best checkpoints
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=SPLIT,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True)

    # Create training batches
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_cols, img_rows),
        batch_size=batch_size,
        subset="training",
        class_mode='categorical',
        shuffle=True)

    for x in train_generator.class_indices:
        print(x)

    # Create testing batches
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_cols, img_rows),
        batch_size=batch_size,
        subset="validation",
        class_mode='categorical',
        shuffle=True)

    print("\n" + "---------------------------------------"
          + "\n" +"Training process commencing..."
          +"\n"+ "---------------------------------------")

    # Start fine-tuning the model
    model.fit_generator(
        train_generator,
        callbacks=[checkpoint, tensorboard],
        verbose=1,
        steps_per_epoch=50,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=10,
        shuffle=True)

    # Save weights when done training
    model.save_weights(model_path)

def main():
    args = get_args()
    data_dir = args.images
    tensorboard = keras.callbacks.TensorBoard(log_dir="drive/app/output/log", write_graph=True, write_images=True)
    img_rows, img_cols = 320, 320  # Resolution of inputs
    num_classes = args.classes
    batch_size = 16
    nb_epoch = 50
    fine_tune_model = InceptionResnetV2(num_classes) # Create a neural network with num_classes output classes
    train(fine_tune_model, img_cols, img_rows, batch_size, data_dir, tensorboard, nb_epoch, args.model)

if __name__ == '__main__':
    main()
