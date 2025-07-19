import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalMaxPooling2D, LeakyReLU, ELU
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc, roc_curve
from sklearn.utils import class_weight
import os
import random
from PIL import Image
import logging
import math


IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_CHANNELS = 3 
BATCH_SIZE = 8
EPOCHS = 30

early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.0000001)
callbacks = [early_stopping, reduce_lr]

base_path = 'butterflies'
df_filename = 'butterflies and moths.csv'

csv_file_path = os.path.join(base_path, df_filename)

def create_dataframe_from_folder(base_folder, df_original_for_mapping):
    image_data = []
    filepath_to_label_map = df_original_for_mapping.set_index('filepaths')['labels'].to_dict()
    filepath_to_dataset_map = df_original_for_mapping.set_index('filepaths')['data set'].to_dict()

    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_folder)

                label = None
                data_set = None

                if relative_path in filepath_to_label_map:
                    label = filepath_to_label_map[relative_path]
                    data_set = filepath_to_dataset_map[relative_path]
                else:
                    parts = relative_path.split(os.sep)
                    if len(parts) >= 3: 
                        data_set_inferred = parts[0]
                        label_inferred = parts[1]
                        label = label_inferred
                        data_set = data_set_inferred 

                if label and data_set: 
                    image_data.append({
                        'filepaths': relative_path,
                        'labels': label,
                        'data set': data_set
                    })
                else:
                    logging.warning(f"Could not infer label/dataset for: {relative_path}. Skipping.")

    return pd.DataFrame(image_data)

df = pd.read_csv(csv_file_path)

train_df = df[df['data set'] == 'train'].copy()
test_df = df[df['data set'] == 'test'].copy()
validation_df = df[df['data set'] == 'valid'].copy()

train_df['image_path'] = train_df['filepaths'].apply(lambda x: os.path.join(base_path, x))
test_df['image_path'] = test_df['filepaths'].apply(lambda x: os.path.join(base_path, x))
validation_df['image_path'] = validation_df['filepaths'].apply(lambda x: os.path.join(base_path, x))

train_df = train_df.rename(columns={'labels': 'label'})
test_df = test_df.rename(columns={'labels': 'label'})
validation_df = validation_df.rename(columns={'labels': 'label'})

print("Dataframes for train, test, and validation created successfully (initial load)!")
print("Train DataFrame shape:", train_df.shape)
print("Test DataFrame shape:", test_df.shape)
print("Validation DataFrame shape:", validation_df.shape)

overall_class_distribution = df['labels'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=overall_class_distribution.index, y=overall_class_distribution.values, palette='cubehelix')
plt.title('Overall Distribution of Classes Across All Data')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

train_class_distribution = train_df['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=train_class_distribution.index, y=train_class_distribution.values, palette='viridis')
plt.title('Distribution of Classes in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

test_class_distribution = test_df['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=test_class_distribution.index, y=test_class_distribution.values, palette='magma')
plt.title('Distribution of Classes in Test Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

validation_class_distribution = validation_df['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=validation_class_distribution.index, y=validation_class_distribution.values, palette='cividis')
plt.title('Distribution of Classes in Validation Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

data_volume = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'Count': [len(train_df), len(test_df), len(validation_df)]
})
plt.figure(figsize=(8, 5))
sns.barplot(x='Dataset', y='Count', data=data_volume, palette='coolwarm')
plt.title('Volume of Train, Test, and Validation Datasets')
plt.xlabel('Dataset Type')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.show()

top_10_classes = overall_class_distribution.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_classes.index, y=top_10_classes.values, palette='Blues_d')
plt.title('Top 10 Most Frequent Classes (Overall)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

bottom_10_classes = overall_class_distribution.tail(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=bottom_10_classes.index, y=bottom_10_classes.values, palette='Reds_d')
plt.title('Bottom 10 Least Frequent Classes (Overall)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

dataset_type_distribution = df['data set'].value_counts()
plt.figure(figsize=(7, 5))
plt.pie(dataset_type_distribution, labels=dataset_type_distribution.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Samples by Dataset Type')
plt.axis('equal')
plt.show()

unique_classes_count = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'Unique Classes': [train_df['label'].nunique(), test_df['label'].nunique(), validation_df['label'].nunique()]
})
plt.figure(figsize=(8, 5))
sns.barplot(x='Dataset', y='Unique Classes', data=unique_classes_count, palette='rocket')
plt.title('Number of Unique Classes per Dataset Split')
plt.xlabel('Dataset Type')
plt.ylabel('Count of Unique Classes')
plt.tight_layout()
plt.show()

def plot_sample_images(dataframe, num_images=9):

    if len(dataframe) < num_images:
        num_images = len(dataframe)
        print(f"Warning: Not enough images to display {num_images}. Displaying all available images.")

    sample_images = dataframe.sample(num_images)
    plt.figure(figsize=(10, 10))
    for i, row in enumerate(sample_images.iterrows()):
        image_path = row[1]['image_path']
        label = row[1]['label']

        if os.path.isfile(image_path):
            try:
                img = Image.open(image_path)
                plt.subplot(3, 3, i + 1)
                plt.imshow(img)
                plt.title(label)
                plt.axis('off')
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                plt.subplot(3, 3, i + 1)
                plt.text(0.5, 0.5, f"Error\n{label}", horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
        else:
            print(f"Skipping {image_path} as it is not a file or does not exist.")
            plt.subplot(3, 3, i + 1)
            plt.text(0.5, 0.5, f"Not Found\n{label}", horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

if 'train_df' in locals() and not train_df.empty:
    plot_sample_images(train_df)
else:
    print("train_df is not loaded or is empty. Please ensure the previous steps were successful.")


NUM_CLASSES = train_df['label'].nunique()
print("Number of images per class in the training set (original):")
print(train_df['label'].value_counts())

min_images_per_class = train_df['label'].value_counts().min()
print(f"\nMinimum number of images in any single class (original): {min_images_per_class}")

max_samples_per_class = train_df['label'].value_counts().max()
print(f"\nTarget samples per class for balancing: {max_samples_per_class}")

save_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

print("\nStarting data augmentation to balance the training set by saving new images to existing folders and updating the original CSV...")
print("!!! WARNING: This will modify your original 'butterflies' folder and 'butterflies and moths.csv' file. !!!")
print("!!! Please ensure you have backed up your data before proceeding. !!!")

for class_name in train_df['label'].unique():
    class_subset = train_df[train_df['label'] == class_name]
    current_count = len(class_subset)

    if current_count < max_samples_per_class:
        num_to_generate = max_samples_per_class - current_count
        print(f"  Class '{class_name}': Current {current_count}, generating {num_to_generate} additional images.")

        example_original_filepath_relative = class_subset['filepaths'].iloc[0]
        target_save_class_dir = os.path.join(base_path, os.path.dirname(example_original_filepath_relative))
        os.makedirs(target_save_class_dir, exist_ok=True) 

        generator_for_saving = save_datagen.flow_from_dataframe(
            dataframe=class_subset,
            x_col='image_path', 
            y_col='label',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=1, 
            class_mode='categorical',
            shuffle=False, 
            save_to_dir=target_save_class_dir, 
            save_prefix='aug',
            seed=random.randint(0, 1000) 
        )

        for i in range(num_to_generate):
            try:
                _ = next(generator_for_saving)
            except Exception as e:
                print(f"Error generating image for class {class_name}: {e}")
                break 

print("\nFinished generating augmented images.")

print(f"\nRebuilding DataFrame by scanning '{base_path}' to include new augmented images...")
df_updated = create_dataframe_from_folder(base_path, df)

print(f"Saving updated DataFrame to original CSV: {csv_file_path}")
df_updated.to_csv(csv_file_path, index=False)
print("CSV file updated successfully.")

df = pd.read_csv(csv_file_path) 

train_df = df[df['data set'] == 'train'].copy()
test_df = df[df['data set'] == 'test'].copy()
validation_df = df[df['data set'] == 'valid'].copy()

train_df['image_path'] = train_df['filepaths'].apply(lambda x: os.path.join(base_path, x))
test_df['image_path'] = test_df['filepaths'].apply(lambda x: os.path.join(base_path, x))
validation_df['image_path'] = validation_df['filepaths'].apply(lambda x: os.path.join(base_path, x))

train_df = train_df.rename(columns={'labels': 'label'})
test_df = test_df.rename(columns={'labels': 'label'})
validation_df = validation_df.rename(columns={'labels': 'label'})


print(f"\nBalanced Train DataFrame shape (after reloading): {train_df.shape}")
print("Balanced Train DataFrame class distribution (after reloading):")
print(train_df['label'].value_counts())

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, 
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

try:
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
except Exception as e:
    logging.error(f"Error loading data generators: {e}")
    raise

print("Class indices from train_generator:", train_generator.class_indices)

def plot_augmented_images(generator, num_images=9):

    images, labels = next(generator)

    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i + 1)
        display_image = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
        plt.imshow(display_image)

        label_index = np.argmax(labels[i])

        class_names = list(generator.class_indices.keys())
        label_name = class_names[label_index]

        plt.title(label_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nDisplaying a batch of augmented training images from the balanced dataset:")
plot_augmented_images(train_generator)


def create_model(base_model_class, input_shape, num_classes):
    model = Sequential()
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    model.add(base_model)

    for layer in base_model.layers:
        layer.trainable = True 

    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5)) 

    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4)) 

    model.add(Dense(num_classes, activation='softmax'))

    return model


logging.info("--- Starting Single-Stage Training for MobileNetV2 Model with Improvements ---")

mobilenetv2 = create_model(MobileNetV2, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), NUM_CLASSES)

total_train_steps = math.ceil(train_generator.samples / BATCH_SIZE) * EPOCHS
lr_schedule = CosineDecay(
    initial_learning_rate=0.0001, 
    decay_steps=total_train_steps,
    alpha=0.01 
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

mobilenetv2.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

mobilenetv2.summary()

class_labels_for_weights = np.unique(train_df['label']) 
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=class_labels_for_weights,
    y=train_df['label']
)
class_weights_dict = dict(zip(class_labels_for_weights, class_weights_array))

mobilenetv2_history = mobilenetv2.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE),
    callbacks=callbacks,
    class_weight=class_weights_dict 
)

mobilenetv2_loss, mobilenetv2_acc = mobilenetv2.evaluate(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))

logging.info(f'MobileNetV2 Test Accuracy: {mobilenetv2_acc:.4f}')
logging.info(f'MobileNetV2 Test Loss: {mobilenetv2_loss:.4f}')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mobilenetv2_history.history['accuracy'], label='Train Accuracy')
plt.plot(mobilenetv2_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mobilenetv2_history.history['loss'], label='Train Loss')
plt.plot(mobilenetv2_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

test_generator.reset()
Y_pred = mobilenetv2.predict(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels)
print("\nClassification Report:\n", report)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()


model_save_path = 'mobilenetv2_butterfly_classifier.keras'
try:
    mobilenetv2.save(model_save_path)
    print(f"\nModel saved successfully to {model_save_path}")
except Exception as e:
    print(f"\nError saving model: {e}")
