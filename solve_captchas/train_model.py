from keras import layers, Sequential
from keras.src.optimizers import Adam
from keras.src.regularizers import l2
from sklearn.model_selection import train_test_split

from utils import read_data, prepare_data


def train_model():
    path_to_letters = "../extracted_letters"
    letters, labels = read_data(path_to_letters)
    letters, one_hot_labels, output_size = prepare_data(labels, letters)
    train_images, test_images, train_labels, test_labels = train_test_split(letters, one_hot_labels, test_size=0.2)

    model = create_model(output_size)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_images, train_labels, batch_size=32, epochs=40)
    show_model_quality(model, test_images, test_labels)
    model.save('sequential.keras')


def create_model(output_size):
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
        layers.Dense(output_size, activation='softmax')
    ])
    return model


def show_model_quality(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'loss: {test_loss}, accuracy: {test_acc}')
    model.summary()


def main():
    train_model()


if __name__ == '__main__':
    main()
