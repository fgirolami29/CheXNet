import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import optax
from sklearn.metrics import roc_auc_score

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
               'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
               'Pleural_Thickening', 'Hernia']

class SimpleDenseNet(nn.Module):
    num_classes: int
    use_running_average: bool  # Specify if we're in training or evaluation mode

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (3, 3), strides=(2, 2))
        x = nn.Dense(self.num_classes)(x)
        x = nn.sigmoid(x)
        return x

def compute_loss(params, batch_stats, model, x, y):
    logits, updated_batch_stats = model.apply({'params': params, 'batch_stats': batch_stats}, x, mutable=['batch_stats'])
    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss, updated_batch_stats

def compute_metrics(logits, labels):
    roc_aucs = []
    for i in range(N_CLASSES):
        roc_aucs.append(roc_auc_score(labels[:, i], logits[:, i]))
    return jnp.mean(jnp.array(roc_aucs))

# Initialize model and optimizer for training
model = SimpleDenseNet(num_classes=N_CLASSES, use_running_average=False)  # Set False for training
rng = random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 224, 224, 3)), mutable=['batch_stats'])
batch_stats = params['batch_stats']
params = params['params']

# Example optimizer setup
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Dummy data for demonstration
x = jnp.ones((64, 224, 224, 3))  # batch of 64 images
y = jnp.zeros((64, N_CLASSES))    # dummy labels

# Forward pass and metric calculation in training mode
logits, updated_batch_stats = model.apply({'params': params, 'batch_stats': batch_stats}, x, mutable=['batch_stats'])
loss, batch_stats = compute_loss(params, batch_stats, model, x, y)
roc_auc = compute_metrics(logits, y)

print("Loss:", loss)
print("Average ROC-AUC:", roc_auc)

# Switch to evaluation mode by setting `use_running_average=True`
eval_model = SimpleDenseNet(num_classes=N_CLASSES, use_running_average=True)
eval_logits = eval_model.apply({'params': params, 'batch_stats': batch_stats}, x)

# Calculate metrics in evaluation mode
eval_roc_auc = compute_metrics(eval_logits, y)
print("Average ROC-AUC in evaluation mode:", eval_roc_auc)
