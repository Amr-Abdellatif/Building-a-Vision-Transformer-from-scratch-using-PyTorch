from pathlib import Path

def get_config():
    """
  This function defines the hyperparameters and configuration settings 
  used for training the PyTorch model.

  Returns:
    A dictionary containing the hyperparameter configuration.
  """
    return {
    # Training parameters
    "batch_size": 64,  # Number of training examples used per update step.
    "pin_memory": True,  # Whether to pin CPU memory for DataLoader (speed improvement).
    "num_workers": 0,    # Number of subprocesses used for data loading (set to 0 for CPU-bound tasks).

    # Data-related parameters
    "img_size": 28,      # Size of the input image.
    "patch_size": 4,      # Patch size for a Vision Transformer model.
    "in_channels": 1,    # Number of input channels for the image (e.g., grayscale: 1, RGB: 3).
    "num_classes": 10,   # Number of classes to predict (e.g., 10 for MNIST dataset).

    # Training hyperparameters
    "lr": 0.001,          # Learning rate for the optimizer.
    "epochs": 5,          # Number of epochs to train the model.

    # Model saving/loading parameters
    "preload": "latest",  # How to preload a model (options: "latest", epoch number, None).
    "model_folder": "model_weights",  # Folder to store/load model weights.
    "model_basename": "t_model_",     # Base name for the saved model files.
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"Vit_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"Vit_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
