from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization,Activation,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import regularizers
num_classes=7
train_dir='/content/drive/MyDrive/ISIC_2018/s_c_train'
val_dir= '/content/drive/MyDrive/ISIC_2018/s_c_validation'
test_dir='/content/drive/MyDrive/ISIC_2018/s_c_test'

size=224
batch_size=32
# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the EfficientNet B0 architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(size, size, 3)))
model.add(BatchNormalization())
model.add(Activation('swish'))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('swish'))

model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('swish'))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('swish'))

model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Define the number of training and validation steps per epoch
num_train_steps = len(train_generator)
num_val_steps = len(val_generator)


epochs=500
# Fit the model using the data generator
from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_steps,
    validation_data=val_generator,
    validation_steps=num_val_steps,
    epochs=epochs,
    verbose=2,callbacks=[earlystopping])

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

#%%

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



# Get the predictions for the validation data

y_pred = model.predict(test_generator).argmax(axis=1)
y_true = test_generator.classes

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
labels = test_generator.class_indices.keys()
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix-test')
plt.show()



# Get the predictions for the training data
train_pred = model.predict(train_generator).argmax(axis=1)
train_true = train_generator.classes

# Compute the confusion matrix for training
train_cm = confusion_matrix(train_true, train_pred)

# Plot the confusion matrix for training
train_labels = train_generator.class_indices.keys()
train_display = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=train_labels)
train_display.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix - Training')
plt.show()

# Get the predictions for the validation data
val_pred = model.predict(val_generator).argmax(axis=1)
val_true = val_generator.classes

# Compute the confusion matrix for validation
val_cm = confusion_matrix(val_true, val_pred)

# Plot the confusion matrix for validation
val_labels = val_generator.class_indices.keys()
val_display = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=val_labels)
val_display.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix - Validation')
plt.show()
#%%
# Save the model for later use
model.save(r'C:\Users\Acer\Desktop\DATAsets\ISIC2018\test_train_val\model_EfficientB0_00.h5')
