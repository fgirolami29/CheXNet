Ecco quattro set di configurazioni per JAX ottimizzati per diverse esigenze: **risparmio risorse**, **standard**, **veloce**, e **ultra**. Puoi scegliere il set in base alle risorse disponibili e all'obiettivo delle tue attività.

---

### **1. Risparmio Risorse**
Questa configurazione minimizza l'uso di risorse GPU e CPU, utile per sistemi con risorse limitate o per lavori che non necessitano di alte prestazioni.

```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disabilita la preallocazione della memoria GPU
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"   # Usa solo il 20% della memoria GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"  # Nessun autotuning
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"       # Permetti una crescita dinamica della memoria GPU
```

- **Uso della memoria GPU:** Minimo
- **Autotuning:** Disabilitato
- **Crescita dinamica della memoria:** Attivata

---

### **2. Standard**
Questa configurazione offre un bilanciamento tra l'uso delle risorse e le prestazioni, adatta per training medio-pesanti senza sovraccaricare il sistema.

```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"   # Abilita la preallocazione
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"   # Usa il 50% della memoria GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"  # Autotuning base
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"      # Disabilita la crescita dinamica della memoria GPU
```

- **Uso della memoria GPU:** Moderato
- **Autotuning:** Livello base
- **Crescita dinamica della memoria:** Disabilitata

---

### **3. Veloce**
Configurazione ottimizzata per alte prestazioni, usa risorse considerevoli ma non satura completamente la GPU.

```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"   # Abilita la preallocazione
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"   # Usa l'80% della memoria GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"  # Autotuning avanzato
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"      # Disabilita la crescita dinamica della memoria GPU
```

- **Uso della memoria GPU:** Alto
- **Autotuning:** Avanzato
- **Crescita dinamica della memoria:** Disabilitata

---

### **4. Ultra**
Configurazione massimizzata per ottenere il massimo delle prestazioni, adatta per training pesanti o sistemi con GPU dedicate.

```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"   # Abilita la preallocazione
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"   # Usa il 100% della memoria GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"  # Autotuning estremo
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"      # Disabilita la crescita dinamica della memoria GPU
```

- **Uso della memoria GPU:** Massimo
- **Autotuning:** Estremo
- **Crescita dinamica della memoria:** Disabilitata

---

### **Come Usare**
Puoi definire una funzione per impostare le configurazioni:

```python
def set_resource_mode(mode="standard"):
    modes = {
        "risparmio": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.2",
            "XLA_FLAGS": "--xla_gpu_autotune_level=0",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        },
        "standard": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
            "XLA_FLAGS": "--xla_gpu_autotune_level=1",
            "TF_FORCE_GPU_ALLOW_GROWTH": "false",
        },
        "veloce": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
            "XLA_FLAGS": "--xla_gpu_autotune_level=2",
            "TF_FORCE_GPU_ALLOW_GROWTH": "false",
        },
        "ultra": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "1.0",
            "XLA_FLAGS": "--xla_gpu_autotune_level=4",
            "TF_FORCE_GPU_ALLOW_GROWTH": "false",
        },
    }
    config = modes.get(mode, modes["standard"])
    for key, value in config.items():
        os.environ[key] = value
    print(f"[INFO] Configurazione impostata su: {mode}")
```

Esempio di utilizzo:

```python
set_resource_mode("ultra")  # Attiva la modalità Ultra
```