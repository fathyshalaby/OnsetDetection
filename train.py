import argparse
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score, recall_score


def train(x_train, y_train, x_val, y_val, x_test, y_test):
    # Define the CNN architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 44, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))

    # Evaluate the model on the test set
    y_pred = model.predict(x_test)
    y_pred_thresh = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    return f1, precision, recall, model
