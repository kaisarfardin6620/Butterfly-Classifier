import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalMaxPooling2D, LeakyReLU
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import class_weight
import os
from PIL import Image
import random
import logging
import math
import datetime
import keras_tuner as kt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_all_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_all_seeds(42)

IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS_PER_TRIAL = 30
STAGE1_EPOCHS_RATIO = 0.3
NUM_FOLDS = 3
EPOCHS = 30

INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

early_stopping_tuner = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min', restore_best_weights=True)
reduce_lr_tuner = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0, mode='min', min_lr=0.0000001)

early_stopping_final = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr_final = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.0000001)

log_dir_tuner_base = "logs/keras_tuner_trials/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

base_path = 'butterflies_v2'
df_filename = 'butterflies and moths.csv'
csv_file_path = os.path.join(base_path, df_filename)

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    logging.error(f"Error: CSV file not found at {csv_file_path}. Please ensure the 'butterflies_v2' directory and 'butterflies and moths.csv' exist.")
    exit()

train_df_original = df[df['data set'] == 'train'].copy()
test_df = df[df['data set'] == 'test'].copy()
validation_df_original = df[df['data set'] == 'valid'].copy()

full_train_val_df = pd.concat([train_df_original, validation_df_original], ignore_index=True)
full_train_val_df['image_path'] = full_train_val_df['filepaths'].apply(lambda x: os.path.join(base_path, x))
full_train_val_df = full_train_val_df.rename(columns={'labels': 'label'})

test_df['image_path'] = test_df['filepaths'].apply(lambda x: os.path.join(base_path, x))
test_df = test_df.rename(columns={'labels': 'label'})

validation_df_original['image_path'] = validation_df_original['filepaths'].apply(lambda x: os.path.join(base_path, x))
validation_df_original = validation_df_original.rename(columns={'labels': 'label'})

logging.info("Dataframes for full_train_val, test, and original validation created successfully!")
logging.info(f"Full Train/Validation DataFrame shape: {full_train_val_df.shape}")
logging.info(f"Test DataFrame shape: {test_df.shape}")
logging.info(f"Original Validation DataFrame shape: {validation_df_original.shape}")

overall_class_distribution = df['labels'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=overall_class_distribution.index, y=overall_class_distribution.values, palette='cubehelix')
plt.title('Overall Distribution of Classes Across All Data')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

train_class_distribution = train_df_original['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=train_class_distribution.index, y=train_class_distribution.values, palette='viridis')
plt.title('Distribution of Classes in Original Training Set')
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

validation_class_distribution = validation_df_original['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=validation_class_distribution.index, y=validation_class_distribution.values, palette='cividis')
plt.title('Distribution of Classes in Original Validation Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

data_volume = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'Count': [len(train_df_original), len(test_df), len(validation_df_original)]
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
    'Unique Classes': [train_df_original['label'].nunique(), test_df['label'].nunique(), validation_df_original['label'].nunique()]
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
        logging.info(f"Warning: Not enough images to display {num_images}. Displaying all available images.")

    sample_images = dataframe.sample(num_images, random_state=42)
    plt.figure(figsize=(10, 10))
    for i, row_data in enumerate(sample_images.iterrows()):
        row = row_data[1]
        image_path = row['image_path']
        label = row['label']

        if os.path.isfile(image_path):
            try:
                img = Image.open(image_path)
                plt.subplot(3, 3, i + 1)
                plt.imshow(img)
                plt.title(label)
                plt.axis('off')
            except Exception as e:
                logging.error(f"Error opening image {image_path}: {e}")
                plt.subplot(3, 3, i + 1)
                plt.text(0.5, 0.5, f"Error\n{label}", horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
        else:
            logging.info(f"Skipping {image_path} as it is not a file or does not exist.")
            plt.subplot(3, 3, i + 1)
            plt.text(0.5, 0.5, f"Not Found\n{label}", horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

if 'full_train_val_df' in locals() and not full_train_val_df.empty:
    plot_sample_images(full_train_val_df)
else:
    logging.info("full_train_val_df is not loaded or is empty. Please ensure the previous steps were successful.")

NUM_CLASSES = full_train_val_df['label'].nunique()
logging.info(f"Number of unique classes: {NUM_CLASSES}")
logging.info("Number of images per class in the full train/validation set:")
logging.info(full_train_val_df['label'].value_counts())

min_images_per_class = full_train_val_df['label'].value_counts().min()
logging.info(f"\nMinimum number of images in any single class: {min_images_per_class}")

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

val_test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_generator_final = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
logging.info(f"Class indices from test_generator_final: {test_generator_final.class_indices}")

final_val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=validation_df_original,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
logging.info(f"Class indices from final_val_generator: {final_val_generator.class_indices}")


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

logging.info("\nDisplaying a batch of augmented training images (from a sample generator):")
sample_train_generator = train_datagen.flow_from_dataframe(
    dataframe=full_train_val_df.sample(n=BATCH_SIZE*2, random_state=42),
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
plot_augmented_images(sample_train_generator)


def build_model(input_shape, num_classes, hps):
    model = Sequential()
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    model.add(base_model)

    units_1 = hps.get('units_1')
    dropout_1 = hps.get('dropout_1')
    l2_1 = hps.get('l2_1')

    units_2 = hps.get('units_2')
    dropout_2 = hps.get('dropout_2')
    l2_2 = hps.get('l2_2')

    units_3 = hps.get('units_3')
    dropout_3 = hps.get('dropout_3')
    l2_3 = hps.get('l2_3')

    model.add(Dense(units_1, kernel_regularizer=regularizers.l2(l2_1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_1))

    model.add(Dense(units_2, kernel_regularizer=regularizers.l2(l2_2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_2))

    model.add(Dense(units_3, kernel_regularizer=regularizers.l2(l2_3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_3))

    model.add(Dense(num_classes, activation='softmax'))

    return model, base_model


class MyHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes, stage1_epochs_ratio, total_epochs_per_trial):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.stage1_epochs_ratio = stage1_epochs_ratio
        self.total_epochs_per_trial = total_epochs_per_trial

    def build(self, hp):
        model, _ = build_model(self.input_shape, self.num_classes, hp)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        base_model = model.layers[0]

        stage1_epochs = max(1, int(self.total_epochs_per_trial * self.stage1_epochs_ratio))
        remaining_epochs = self.total_epochs_per_trial - stage1_epochs

        base_model.trainable = False
        stage1_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.get('learning_rate_stage1'))
        loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=hp.get('label_smoothing'))
        model.compile(optimizer=stage1_optimizer, loss=loss_function, metrics=['accuracy'])
        logging.info(f"Trial {hp.trial_id} - Stage 1: Training custom head layers for {stage1_epochs} epochs.")
        history_stage1 = model.fit(epochs=stage1_epochs, *args, **kwargs)

        combined_history = {}
        for k, v in history_stage1.history.items():
            combined_history[k] = v

        if remaining_epochs > 0:
            base_model.trainable = True
            stage2_initial_lr = hp.get('learning_rate_stage2')

            if hasattr(kwargs.get('x'), 'samples') and hasattr(kwargs.get('x'), 'batch_size'):
                steps_per_epoch_train = math.ceil(kwargs['x'].samples / kwargs['batch_size'])
            else:
                logging.warning("Could not determine steps_per_epoch from generator. Using default or assuming dataset size. This might affect CosineDecay steps.")
                steps_per_epoch_train = kwargs.get('steps_per_epoch', 1)

            total_train_steps_stage2 = steps_per_epoch_train * remaining_epochs

            stage2_lr_schedule = CosineDecay(
                initial_learning_rate=stage2_initial_lr,
                decay_steps=total_train_steps_stage2,
                alpha=0.01
            )
            stage2_optimizer = tf.keras.optimizers.Adam(learning_rate=stage2_lr_schedule)
            model.compile(optimizer=stage2_optimizer, loss=loss_function, metrics=['accuracy'])
            logging.info(f"Trial {hp.trial_id} - Stage 2: Fine-tuning entire model for {remaining_epochs} epochs.")
            history_stage2 = model.fit(epochs=self.total_epochs_per_trial, initial_epoch=stage1_epochs, *args, **kwargs)

            for k in history_stage2.history.keys():
                combined_history[k] = combined_history.get(k, []) + history_stage2.history[k]
        else:
            logging.info(f"Trial {hp.trial_id} - Stage 2 skipped as remaining epochs is 0.")

        return combined_history


logging.info("--- Starting Nested Hyperparameter Tuning with Stratified K-Fold ---")

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_results = []
best_hps_per_fold = []
best_models_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_val_df['image_path'], full_train_val_df['label'])):
    logging.info(f"\n--- Starting Fold {fold + 1}/{NUM_FOLDS} ---")

    train_fold_df = full_train_val_df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = full_train_val_df.iloc[val_idx].reset_index(drop=True)

    train_generator_fold = train_datagen.flow_from_dataframe(
        dataframe=train_fold_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_generator_fold = val_test_datagen.flow_from_dataframe(
        dataframe=val_fold_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_labels_for_weights_fold = np.unique(train_fold_df['label'])
    class_weights_array_fold = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=class_labels_for_weights_fold,
        y=train_fold_df['label']
    )
    class_weights_dict_fold = dict(zip(class_labels_for_weights_fold, class_weights_array_fold))

    tuner_log_dir = os.path.join(log_dir_tuner_base, f"fold_{fold + 1}")
    tensorboard_callback_tuner = TensorBoard(log_dir=tuner_log_dir, histogram_freq=1)

    tuner_callbacks = [early_stopping_tuner, reduce_lr_tuner, tensorboard_callback_tuner]

    tuner = kt.Hyperband(
        MyHyperModel(
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            stage1_epochs_ratio=STAGE1_EPOCHS_RATIO,
            total_epochs_per_trial=EPOCHS_PER_TRIAL
        ),
        objective='val_accuracy',
        max_epochs=EPOCHS_PER_TRIAL,
        factor=3,
        directory='keras_tuner_results',
        project_name=f'butterfly_classification_fold_{fold + 1}',
        overwrite=True
    )

    logging.info(f"Starting Hyperparameter search for Fold {fold + 1}...")
    tuner.search(
        train_generator_fold,
        validation_data=val_generator_fold,
        callbacks=tuner_callbacks,
        class_weight=class_weights_dict_fold
    )

    best_hps_fold = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model_fold = tuner.get_best_models(num_models=1)[0]

    logging.info(f"Best Hyperparameters for Fold {fold + 1}:")
    logging.info(best_hps_fold.values)

    fold_val_loss, fold_val_acc = best_model_fold.evaluate(val_generator_fold, steps=math.ceil(val_generator_fold.samples / BATCH_SIZE))
    logging.info(f"Fold {fold + 1} Validation Accuracy: {fold_val_acc:.4f}")
    logging.info(f"Fold {fold + 1} Validation Loss: {fold_val_loss:.4f}")

    fold_results.append({'fold': fold + 1, 'val_accuracy': fold_val_acc, 'val_loss': fold_val_loss, 'best_hps': best_hps_fold.values})
    best_models_per_fold.append(best_model_fold)
    best_hps_per_fold.append(best_hps_fold)


logging.info("\n--- K-Fold Cross-Validation Results Summary ---")
for result in fold_results:
    logging.info(f"Fold {result['fold']}: Val Accuracy = {result['val_accuracy']:.4f}, Val Loss = {result['val_loss']:.4f}")
    logging.info(f"  Best HPs: {result['best_hps']}")

avg_val_accuracy = np.mean([res['val_accuracy'] for res in fold_results])
avg_val_loss = np.mean([res['val_loss'] for res in fold_results])
logging.info(f"\nAverage Validation Accuracy across {NUM_FOLDS} folds: {avg_val_accuracy:.4f}")
logging.info(f"Average Validation Loss across {NUM_FOLDS} folds: {avg_val_loss:.4f}")


best_fold_index = np.argmax([res['val_accuracy'] for res in fold_results])
overall_best_hps = best_hps_per_fold[best_fold_index]
logging.info(f"\n--- Training Final Model with Best Hyperparameters from Fold {best_fold_index + 1} ---")
logging.info(f"Overall Best Hyperparameters: {overall_best_hps.values}")

final_model, final_base_model = build_model(INPUT_SHAPE, NUM_CLASSES, overall_best_hps)

final_class_labels_for_weights = np.unique(full_train_val_df['label'])
final_class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=final_class_labels_for_weights,
    y=full_train_val_df['label']
)
final_class_weights_dict = dict(zip(final_class_labels_for_weights, final_class_weights_array))

final_train_val_generator = train_datagen.flow_from_dataframe(
    dataframe=full_train_val_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

final_label_smoothing = overall_best_hps.get('label_smoothing')
final_loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=final_label_smoothing)

final_total_epochs = EPOCHS
final_stage1_epochs = max(1, int(final_total_epochs * STAGE1_EPOCHS_RATIO))
final_remaining_epochs = final_total_epochs - final_stage1_epochs

final_base_model.trainable = False
final_stage1_optimizer = tf.keras.optimizers.Adam(learning_rate=overall_best_hps.get('learning_rate_stage1'))
final_model.compile(optimizer=final_stage1_optimizer, loss=final_loss_function, metrics=['accuracy'])
logging.info(f"Final Model - Stage 1: Training custom head layers for {final_stage1_epochs} epochs.")
final_history_stage1 = final_model.fit(
    final_train_val_generator,
    steps_per_epoch=math.ceil(final_train_val_generator.samples / BATCH_SIZE),
    epochs=final_stage1_epochs,
    validation_data=final_val_generator,
    validation_steps=math.ceil(final_val_generator.samples / BATCH_SIZE),
    callbacks=[early_stopping_final, reduce_lr_final],
    class_weight=final_class_weights_dict
)

final_history_combined = {}
for key, value in final_history_stage1.history.items():
    final_history_combined[key] = value

if final_remaining_epochs > 0:
    final_base_model.trainable = True
    final_stage2_initial_lr = overall_best_hps.get('learning_rate_stage2')

    final_total_train_steps_stage2 = math.ceil(final_train_val_generator.samples / BATCH_SIZE) * final_remaining_epochs
    final_stage2_lr_schedule = CosineDecay(
        initial_learning_rate=final_stage2_initial_lr,
        decay_steps=final_total_train_steps_stage2,
        alpha=0.01
    )
    final_stage2_optimizer = tf.keras.optimizers.Adam(learning_rate=final_stage2_lr_schedule)
    final_model.compile(optimizer=final_stage2_optimizer, loss=final_loss_function, metrics=['accuracy'])
    logging.info(f"Final Model - Stage 2: Fine-tuning entire model for {final_remaining_epochs} epochs.")
    final_history_stage2 = final_model.fit(
        final_train_val_generator,
        steps_per_epoch=math.ceil(final_train_val_generator.samples / BATCH_SIZE),
        epochs=final_total_epochs,
        initial_epoch=final_stage1_epochs,
        validation_data=final_val_generator,
        validation_steps=math.ceil(final_val_generator.samples / BATCH_SIZE),
        callbacks=[early_stopping_final, reduce_lr_final],
        class_weight=final_class_weights_dict
    )

    for key in final_history_stage2.history.keys():
        final_history_combined[key] = final_history_combined.get(key, []) + final_history_stage2.history[key]
else:
    logging.info("Final Model - Stage 2 skipped as remaining epochs is 0.")


final_model_loss, final_model_acc = final_model.evaluate(test_generator_final, steps=math.ceil(test_generator_final.samples / BATCH_SIZE))

logging.info(f'Final Model Test Accuracy: {final_model_acc:.4f}')
logging.info(f'Final Model Test Loss: {final_model_loss:.4f}')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(final_history_combined['accuracy'], label='Train Accuracy')
plt.plot(final_history_combined['val_accuracy'], label='Validation Accuracy')
plt.title('Final Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(final_history_combined['loss'], label='Train Loss')
plt.plot(final_history_combined['val_loss'], label='Validation Loss')
plt.title('Final Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

test_generator_final.reset()
Y_pred = final_model.predict(test_generator_final, steps=math.ceil(test_generator_final.samples / BATCH_SIZE))
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true_classes = test_generator_final.classes

class_labels = list(test_generator_final.class_indices.keys())
report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels)
logging.info(f"\nFinal Model Classification Report:\n{report}")

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Final Model Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

model_save_path = 'final_tuned_mobilenetv2_butterfly_classifier.keras'
try:
    final_model.save(model_save_path)
    logging.info(f"\nFinal Model saved successfully to {model_save_path}")
except Exception as e:
    logging.error(f"\nError saving final model: {e}")
