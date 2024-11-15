Sì, è possibile visualizzare in tempo reale l'avanzamento e il tempo trascorso all'ultima riga della shell utilizzando una barra di progresso. La libreria `tqdm` permette di aggiornare dinamicamente il progresso e mostrare il tempo impiegato, rimanendo sempre nell’ultima riga. 

Di seguito trovi un esempio di come integrare `tqdm` con un timer e la percentuale di completamento per un processo di caricamento e addestramento:

```python
from tqdm import tqdm
import time
import os

# Impostazioni di esempio
EPOCHS = 5
BATCH_SIZE = 8
TOTAL_IMAGES = 1000  # Numero totale di immagini, per il calcolo della percentuale

# Funzione simulata di caricamento e preprocessamento delle immagini
def load_data(dataset, total_images=TOTAL_IMAGES):
    success_count = 0
    missing_count = 0
    
    # Barra di progresso per il caricamento delle immagini
    with tqdm(total=total_images, desc="Caricamento immagini", unit="img", dynamic_ncols=True) as pbar:
        for i in range(total_images):
            # Simulazione del caricamento dell'immagine
            time.sleep(0.01)  # Simula il tempo di caricamento
            if i % 10 == 0:
                missing_count += 1
            else:
                success_count += 1

            # Aggiornamento della barra di progresso
            pbar.update(1)
            pbar.set_postfix(success=success_count, missing=missing_count)

    print(f"\n[INFO] Totale immagini caricate: {success_count}")
    print(f"[INFO] Totale immagini mancanti: {missing_count}")
    return success_count, missing_count

# Funzione simulata di training
def train_model(epochs, total_batches=100):
    with tqdm(total=epochs * total_batches, desc="Training Model", unit="batch", dynamic_ncols=True) as pbar:
        for epoch in range(epochs):
            for batch in range(total_batches):
                # Simulazione del tempo di addestramento per batch
                time.sleep(0.05)
                
                # Aggiornamento della barra di progresso per ogni batch
                pbar.update(1)
                pbar.set_description(f"Training EPOCH {epoch + 1}/{epochs}")
                pbar.set_postfix(batch=batch + 1, epoch=epoch + 1)

# Simulazione del caricamento dati e del training
dataset = "path/to/dataset"  # Inserisci il percorso al dataset
success, missing = load_data(dataset)

# Avvio dell'addestramento
train_model(EPOCHS)
```

### Come funziona:
- **`tqdm(total=...)`**: Imposta il numero totale di iterazioni da completare per la barra di progresso.
- **`pbar.update(1)`**: Aggiorna la barra per ogni immagine/batch elaborato.
- **`pbar.set_description()`** e **`pbar.set_postfix()`**: Mostrano informazioni dinamiche come il numero di `epoch`, `batch`, il numero di immagini caricate con successo e quelle mancanti.

### Vantaggi:
- L'output rimane sempre sull'ultima riga, aggiornandosi automaticamente.
- Visualizza una stima del tempo rimanente e del tempo totale impiegato.
- È possibile vedere la percentuale di completamento in tempo reale.

Questo ti permette di avere un feedback costante e visivo durante l’esecuzione del codice, ideale per il monitoraggio di operazioni lunghe e intensive.