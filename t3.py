from PIL import Image
import os
import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from tqdm import tqdm
import time
from rich import print

# Parametri
N_CLASSES = 14
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5
# Imposta il percorso del checkpoint come assoluto
CHECKPOINT_PATH = os.path.abspath('./models/j4x/checkpoint/')

IMAGES_PATH = './ChestX-ray14/images0'
TRAIN_LIST = './ChestX-ray14/labels/train_list_j4x.txt'

# Definizione delle classi del dataset
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# 1. Caricamento e Preprocessamento dei Dati
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 0.5) * 2  # Normalizza tra -1 e 1
    return image, label

class CustomDataset:
    def __init__(self, image_list_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = []
        self.labels = []
        self.success_count = 0  # Contatore delle immagini caricate
        self.missing_count = 0  # Contatore delle immagini mancanti
        
        with open(image_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                self.image_list.append(parts[0])
                self.labels.append([float(x) for x in parts[1:]])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = os.path.join(self.data_dir, self.image_list[index])
        try:
            image = Image.open(image_name).convert('RGB')
            self.success_count += 1  # Incrementa se l'immagine viene caricata correttamente
        except FileNotFoundError:
            print(f"[red]Immagine mancante: {image_name}. Skipping.[/red]")
            self.missing_count += 1  # Incrementa se l'immagine manca
            return None  # Escludi l'immagine non trovata

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        return image, label
    
    
# Funzione per creare il percorso del checkpoint se non esiste
def ensure_checkpoint_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[bold green]:floppy_disk: Creato percorso del checkpoint: {path}[/bold green]")
    else:
        print(f"[bold cyan]:check_mark: Percorso del checkpoint già esistente: {path}[/bold cyan]")

def load_data_from_dataset(dataset):
    images, labels = [], []
    total_start_time = time.time()  # Avvio del cronometro totale

    # Barra di progresso per il caricamento delle immagini
    with tqdm(total=len(dataset), desc="Caricamento immagini", unit="img", dynamic_ncols=True) as pbar:
        for i in range(len(dataset)):
            start_time = time.time()  # Avvio del cronometro per ogni immagine
            result = dataset[i]
            
            if result is not None:
                image, label = result
                image, label = preprocess_image(image, label)
                images.append(image)
                labels.append(label)
                
                # Aggiornamento delle informazioni sulla barra di progresso
                pbar.set_postfix(
                    tempo=f"{time.time() - start_time:.4f}s",
                    caricate=dataset.success_count,
                    mancanti=dataset.missing_count
                )
            else:
                # Aggiorna la barra anche se l'immagine manca
                pbar.set_postfix(mancanti=dataset.missing_count)
            
            # Aggiorna la barra di progresso
            pbar.update(1)

    total_duration = time.time() - total_start_time  # Durata totale
    print(f"[bold green]Totale immagini caricate: {dataset.success_count}[/bold green]")
    print(f"[bold red]Totale immagini mancanti: {dataset.missing_count}[/bold red]")
    print(f"[bold yellow]Tempo totale di caricamento: {total_duration:.2f} secondi[/bold yellow]")

    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)


# 2. Definizione del Modello (senza DenseNet121)
class SimpleCheXNet(nn.Module):
    num_classes: int

    def setup(self):
        self.dense1 = nn.Dense(512)
        self.dense2 = nn.Dense(256)
        self.out_layer = nn.Dense(self.num_classes)
        self.dropout = nn.Dropout(0.5)

    def __call__(self, x, train=True):
        x = x.reshape((x.shape[0], -1))  # Flattening input
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.dense2(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.out_layer(x)
        return nn.sigmoid(x)

# 3. Funzioni per lo stato di training, perdita e aggiornamento
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

# 4. Training Loop
def train_and_save(model, train_dataset, rng, epochs=EPOCHS, checkpoint_path=CHECKPOINT_PATH):
    ensure_checkpoint_path(checkpoint_path)
    state = create_train_state(rng, model, LEARNING_RATE)
    state = checkpoints.restore_checkpoint(checkpoint_path, state)
    
    start_time = time.time()  # Avvio del timer totale
    for epoch in range(epochs):
        print(f"[bold yellow]:hourglass: Inizio EPOCH {epoch + 1}[/bold yellow]")
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
        print(f"[bold green]:white_check_mark: Fine EPOCH {epoch + 1} - Tempo totale: {epoch_duration:.4f} secondi - Loss media EPOCH: {total_loss / len(train_dataset):.4f}[/bold green]")
        checkpoints.save_checkpoint(checkpoint_path, state, epoch + 1)

    total_duration = time.time() - start_time  # Calcolo della durata totale
    print(f"[bold green]:tada: Training completo in {total_duration / 60:.2f} minuti e checkpoint salvato.[/bold green]")

# 5. Esecuzione del training
rng = jax.random.PRNGKey(0)
model = SimpleCheXNet(num_classes=N_CLASSES)

# Utilizzo del codice
dataset = CustomDataset(image_list_file=TRAIN_LIST, data_dir=IMAGES_PATH)
train_dataset = load_data_from_dataset(dataset)

train_and_save(model, train_dataset, rng)
