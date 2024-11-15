import os
import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import time
from rich import print

# Parametri
N_CLASSES = 14
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5
CHECKPOINT_PATH = os.path.abspath('./j4x/checkpoint/')  # Convertito in percorso assoluto
IMAGES_PATH = './ChestX-ray14/images'
TRAIN_LIST = './ChestX-ray14/labels/train_list.txt'

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# Funzione per creare il percorso del checkpoint se non esiste
def ensure_checkpoint_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[bold green]:floppy_disk: Creato percorso del checkpoint: {path}[/bold green]")
    else:
        print(f"[bold cyan]:check_mark: Percorso del checkpoint già esistente: {path}[/bold cyan]")

# Caricamento e preprocessamento dei dati
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 0.5) * 2
    return image, label

def load_data_from_txt(file_path, image_dir):
    images = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_path = os.path.join(image_dir, parts[0])
            label = [float(x) for x in parts[1:]]
            try:
                start_time = time.time()
                image = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
                image, label = preprocess_image(image, label)
                images.append(image)
                labels.append(label)
                print(f"[green]:open_file_folder: Caricata immagine: {img_path}[/green], [cyan]Tempo di caricamento: {time.time() - start_time:.4f} secondi[/cyan]")
            except:
                print(f"[red]:x: [ERRORE] Immagine non trovata: {img_path}[/red]")
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)

# Definizione del Modello
class SimpleCheXNet(nn.Module):
    num_classes: int

    def setup(self):
        self.dense1 = nn.Dense(512)
        self.dense2 = nn.Dense(256)
        self.out_layer = nn.Dense(self.num_classes)
        self.dropout = nn.Dropout(0.5)

    def __call__(self, x, train=True):
        x = x.reshape((x.shape[0], -1))
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.dense2(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.out_layer(x)
        return nn.sigmoid(x)

def binary_cross_entropy_loss(logits, labels):
    return -jnp.mean(labels * jnp.log(logits + 1e-7) + (1 - labels) * jnp.log(1 - logits + 1e-7))

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, *IMG_SIZE, 3)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, dropout_rng):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'], train=True, rngs={"dropout": dropout_rng})
        loss = binary_cross_entropy_loss(logits, batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_and_save(model, train_dataset, rng, epochs=EPOCHS, checkpoint_path=CHECKPOINT_PATH):
    ensure_checkpoint_path(checkpoint_path)
    
    state = create_train_state(rng, model, LEARNING_RATE)
    state = checkpoints.restore_checkpoint(checkpoint_path, state)

    for epoch in range(epochs):
        print(f"[bold yellow]:hourglass: Inizio epoca {epoch + 1}[/bold yellow]")
        epoch_start_time = time.time()
        total_loss = 0

        for i, batch in enumerate(train_dataset):
            batch_start_time = time.time()
            images, labels = batch
            batch = {'image': jnp.array(images), 'label': jnp.array(labels)}
            
            # Crea un nuovo dropout_rng per ogni batch
            rng, dropout_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, dropout_rng)
            
            total_loss += loss
            print(f"[blue]Batch {i + 1} - Tempo: {time.time() - batch_start_time:.4f} secondi[/blue] - [magenta]Loss batch: {loss:.4f}[/magenta]")

        epoch_duration = time.time() - epoch_start_time
        print(f"[bold green]:white_check_mark: Fine epoca {epoch + 1} - Tempo totale: {epoch_duration:.4f} secondi - Loss media epoca: {total_loss / len(train_dataset):.4f}[/bold green]")
        checkpoints.save_checkpoint(checkpoint_path, state, epoch + 1)

    print("[bold green]:tada: Training completo e checkpoint salvato.[/bold green]")

rng = jax.random.PRNGKey(0)
model = SimpleCheXNet(num_classes=N_CLASSES)
train_dataset = load_data_from_txt(TRAIN_LIST, IMAGES_PATH)

train_and_save(model, train_dataset, rng)
