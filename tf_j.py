import tensorflow as tf
import jax.numpy as jnp

DATA_DIR = './ChestX-ray14/images'
LABELS_FILE = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)

def load_image(filepath, label):
    """Load and preprocess an image."""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0  # Normalize to [0,1]
    return image, label

def create_dataset(data_dir, labels_file, batch_size):
    """Create a dataset with images and labels from the specified directory and labels file."""
    # Read the labels file (assuming a format like "image_path label")
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    filepaths, labels = zip(*[line.strip().split() for line in lines])
    filepaths = [f"{data_dir}/{path}" for path in filepaths]
    labels = [int(label) for label in labels]

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create the dataset
train_dataset = create_dataset(DATA_DIR, LABELS_FILE, BATCH_SIZE)

# Convert a batch to JAX
for images, labels in train_dataset:
    images = jnp.array(images)  # Convert TensorFlow tensors to JAX numpy arrays
    labels = jnp.array(labels)  # Convert labels as well
    break  # For demonstration purposes, we just take one batch
