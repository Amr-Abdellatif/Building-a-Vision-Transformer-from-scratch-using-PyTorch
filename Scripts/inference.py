import torch
from torchvision import transforms
from architecture import ViT
from config import *

def inference(config):

    # Define the path to your saved model file (replace with your actual path)
    model_filename = "path to your model"
    
    # Load the model state dictionary
    checkpoint = torch.load(model_filename)
    model_state_dict = checkpoint["model_state_dict"]

    # Create a new instance of the vision transformer model (replace with your actual class)
    model = ViT(num_classes=config["num_classes"],
                in_channels=config["in_channels"],
                patch_size=config["patch_size"],
                img_size=config["img_size"])

    # Load the model state dictionary into the new model
    model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode (recommended for inference)
    model.eval()

    # Define data transformations for MNIST images (assuming grayscale)
    transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),  # Resize to expected size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST data (assuming grayscale)
    ])

    # Function to load your PIL image object (replace with your implementation)
    def load_image(path):
        try:
            from PIL import Image  # Assuming you're using PIL for image loading
            return Image.open(path).convert('L')  # Convert to grayscale if needed
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    # Load your image using the load_image function (replace with your image path)
    input_image_path = "path to test image"
    input_image = load_image(input_image_path)

    # Check if image was loaded successfully
    if input_image is None:
        print("Failed to load image. Exiting...")
        return

    # Convert image to tensor and add batch dimension
    input_data = transform(input_image).unsqueeze(0)

    # Pass the input data through the model for inference
    output = model(input_data)

    # Process the output (classification for MNIST)
    predicted_class = torch.argmax(output)
    print(f"Predicted digit: {predicted_class}")

if __name__ == "__main__":
    config = get_config()
    inference(config)
