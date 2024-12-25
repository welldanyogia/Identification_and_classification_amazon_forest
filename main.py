from models.evaluate import plot_metrics
from train.train_cnn import train_cnn as train_cnn
from train.train_mobilenet import train_mobilenet
from train.train_resnet import train_resnet as train_resnet
from models.evaluate import evaluate_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    # Example input shape and number of classes
    input_shape = (224, 224, 3)  # Change as necessary
    num_classes = 10  # Adjust according to your classes

    # Create a data generator for testing
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_data = test_datagen.flow_from_directory(
        'data/planets/test-jpg',  # Specify the correct path to your test data
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='categorical'
    )

    print("Starting training for CNN model...")
    cnn_model, cnn_history = train_cnn(epochs=15, batch_size=32)

    print("\nEvaluating CNN model...")
    evaluate_model(cnn_model, test_data)

    # Optionally, plot metrics for CNN model
    plot_metrics(cnn_history, metric='accuracy',name='CNN model')
    plot_metrics(cnn_history, metric='loss', name='CNN model')

    # print("\nStarting training for ResNet model...")
    # resnet_model, resnet_history = train_resnet(epochs=15, batch_size=32)
    #
    # print("\nEvaluating ResNet model...")
    # evaluate_model(resnet_model, test_data)
    #
    # # Optionally, plot metrics for ResNet model
    # plot_metrics(resnet_history, metric='accuracy', name='ResNet model')
    # plot_metrics(resnet_history, metric='loss', name='ResNet model')
    #
    # print("Starting training for MobileNet model...")
    # mobilenet_model, mobilenet_history = train_mobilenet(input_shape=input_shape, epochs=15,
    #                                                      batch_size=32)
    #
    # print("\nEvaluating MobileNet model...")
    # evaluate_model(mobilenet_model, test_data)
    #
    # # Optionally, plot metrics for MobileNet model
    # plot_metrics(mobilenet_history, metric='accuracy')
    # plot_metrics(mobilenet_history, metric='loss')


if __name__ == "__main__":
    main()
