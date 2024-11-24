import pydicom  # Importing the pydicom library for handling DICOM files
from pydicom import dcmread  # Importing the dcmread function from pydicom to read DICOM files
import os  # Importing the os module for operating system related functionalities
from os import listdir  # Importing listdir function from os module to list files in a directory
from os.path import isfile, join  # Importing isfile and join functions from os.path module for file-related operations
import torch  # Importing PyTorch library
import torch.nn as nn  # Importing neural network module from PyTorch
from torch.utils.data import TensorDataset, DataLoader  # Importing TensorDataset and DataLoader from PyTorch for handling datasets
import matplotlib.pyplot as plt  # Importing matplotlib for visualization
import torchmetrics  # Importing torchmetrics library for evaluation metrics
from torchmetrics.classification import BinaryAccuracy  # Importing BinaryAccuracy class from torchmetrics for binary classification
import numpy as np  # Importing NumPy library for numerical computations
from torch.optim import Adam  # Importing Adam optimizer from PyTorch
from tqdm import tqdm  # Importing tqdm for progress bars


class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = self.conv_block(in_channels, 16, 3, 1, 1)
    self.maxpool1 = self.maxpool_block(2, 2, 0)
    self.conv2 = self.conv_block(16, 32, 3, 1, 1)
    self.maxpool2 = self.maxpool_block(2, 2, 0)
    self.conv3 = self.conv_block(32, 64, 3, 1, 1)
    self.maxpool3 = self.maxpool_block(2, 2, 0)
    self.middle = self.conv_block(64, 128, 3, 1, 1)
    self.upsample3 = self.transposed_block(128, 64, 3, 2, 1, 1)
    self.upconv3 = self.conv_block(128, 64, 3, 1, 1)
    self.upsample2 = self.transposed_block(64, 32, 3, 2, 1, 1)
    self.upconv2 = self.conv_block(64, 32, 3, 1, 1)
    self.upsample1 = self.transposed_block(32, 16, 3, 2, 1, 1)
    self.upconv1 = self.conv_block(32, 16, 3, 1, 1)
    self.final = self.final_layer(16, 1, 1, 1, 0)

  def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
    convolution = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True))
    return convolution

  def maxpool_block(self, kernel_size, stride, padding):
      maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
      nn.Dropout2d(0.5))
      return maxpool

  def transposed_block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
      transposed = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
      padding=padding, output_padding=output_padding)
      return transposed

  def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
      final = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
      return final

  def forward(self, x):
      # downsampling part
      conv1 = self.conv1(x)
      maxpool1 = self.maxpool1(conv1)
      conv2 = self.conv2(maxpool1)
      maxpool2 = self.maxpool2(conv2)
      conv3 = self.conv3(maxpool2)
      maxpool3 = self.maxpool3(conv3)
      # middle part
      middle = self.middle(maxpool3)
      # upsampling part
      upsample3 = self.upsample3(middle)
      upconv3 = self.upconv3(torch.cat([upsample3, conv3], 1))
      upsample2 = self.upsample2(upconv3)
      upconv2 = self.upconv2(torch.cat([upsample2, conv2], 1))
      upsample1 = self.upsample1(upconv2)
      upconv1 = self.upconv1(torch.cat([upsample1, conv1], 1))
      final_layer = self.final(upconv1)
      return final_layer
  

# Function to convert DICOM files to numpy array
def dicom_to_numpy(dicom_dir):
    """
    Convert DICOM files to a 3D numpy array.

    Args:
    - dicom_dir (str): Path to the directory containing DICOM files.

    Returns:
    - volume (numpy.ndarray): 3D numpy array representing the DICOM volume.
    """
    # Read DICOM files into a list of DICOM slices
    dicom_slices = [dcmread(os.path.join(dicom_dir, filename)) for filename in os.listdir(dicom_dir)]
    # Sort DICOM slices based on the z-coordinate (slice position)
    dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # Extract pixel arrays from DICOM slices
    pixel_arrays = [slice.pixel_array for slice in dicom_slices]
    # Stack pixel arrays to form a 3D volume
    volume = np.stack(pixel_arrays, axis=-1)
    return volume

def prepare_images_tensor(dataset_dir):
    """
    Prepare a PyTorch tensor containing image data from DICOM files.

    Args:
    - dataset_dir (str): Path to the directory containing DICOM datasets.

    Returns:
    - images_tensor (torch.Tensor): PyTorch tensor containing image data.
    """
    # List of case directories
    case_dirs = [os.path.join(dataset_dir, f"Case_{str(i).zfill(3)}") for i in range(0, 12)]
    # Convert DICOM datasets to numpy arrays
    case_volumes = []
    for case_dir in case_dirs:
        case_volume = dicom_to_numpy(case_dir)
        case_volumes.append(case_volume)
    # Reshape each case volume to (512, 512) format
    reshaped_slices = []
    for case_volume in case_volumes:
        num_slices = case_volume.shape[-1]  # Get the number of slices
        for i in range(num_slices):
            reshaped_slices.append(case_volume[:, :, i])
    # Combine all reshaped slices into a single array
    combined_slices = np.stack(reshaped_slices, axis=0)
    # Convert combined_slices to a PyTorch tensor with data type torch.float32
    images_tensor = torch.from_numpy(combined_slices.astype(np.float32))
    return images_tensor

def prepare_masks_tensor(dataset_dir):
    """
    Prepare a PyTorch tensor containing mask data from segmentation files.

    Args:
    - dataset_dir (str): Path to the directory containing segmentation files.

    Returns:
    - masks_tensor (torch.Tensor): PyTorch tensor containing mask data.
    """
    # List to store all the loaded 3D arrays
    masks_all = []
    # Iterate over each file
    for i in range(12):
        # Construct the file path for each case
        image_file = os.path.join(dataset_dir, f"Case_{str(i).zfill(3)}_seg.npz")
        # Load segmentation masks from the NPZ file
        segmentation_data = np.load(image_file)
        image_array = segmentation_data['masks']
        for j in range(len(image_array)):
            image_array_j = image_array[j,:,:]
            # Append the 3D array to the list
            masks_all.append(image_array_j)
    # Combine all reshaped slices into a single array
    masks_all = np.stack(masks_all, axis=0)
    # Convert combined_slices to a PyTorch tensor with data type torch.float32
    masks_tensor = torch.from_numpy(masks_all.astype(np.float32))
    return masks_tensor

def split_dataset(images_tensor, masks_tensor):
    """
    Split the dataset into train and test sets.

    Args:
        images_tensor (torch.Tensor): Tensor containing images.
        masks_tensor (torch.Tensor): Tensor containing corresponding masks.

    Returns:
        tuple: A tuple containing train_images, train_masks, test_images, and test_masks.
    """
    train_samples = 1151
    # Split the tensors into train and test sets
    train_images = images_tensor[:train_samples]
    train_masks = masks_tensor[:train_samples]
    test_images = images_tensor[train_samples:]
    test_masks = masks_tensor[train_samples:]
    # Add a new channel dimension
    train_images = train_images.unsqueeze(1)
    train_masks = train_masks.unsqueeze(1)
    test_images = test_images.unsqueeze(1)
    test_masks = test_masks.unsqueeze(1)
    return train_images, train_masks, test_images, test_masks

def soft_dice_loss(pred, target, smooth=1e-5):
    """
    Calculate the soft dice loss.

    Args:
        pred (torch.Tensor): Predicted output.
        target (torch.Tensor): Ground truth.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Soft dice loss.
    """
    intersection = torch.sum(pred * target)
    dice_coeff = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    dice_loss = 1. - dice_coeff
    return dice_loss

def custom_loss(pred, target, alpha=0.5):
    """
    Calculate the custom loss which is a combination of binary cross-entropy and soft dice loss.

    Args:
        pred (torch.Tensor): Predicted output.
        target (torch.Tensor): Ground truth.
        alpha (float): Weight parameter for balancing binary cross-entropy and soft dice loss.

    Returns:
        torch.Tensor: Custom loss.
    """
    bce_loss = torch.nn.BCEWithLogitsLoss()(pred, target)
    dice_loss = soft_dice_loss(torch.sigmoid(pred), target)
    total_loss = alpha * bce_loss + (1 - alpha) * dice_loss
    return total_loss

def training(model, train_images, train_masks, device, epochs=10, learning_rate=0.1):
    """
    Train the model.

    Args:
        model (torch.nn.Module): Neural network model.
        train_images (torch.Tensor): Tensor containing training images.
        train_masks (torch.Tensor): Tensor containing corresponding training masks.
        device (torch.device): Device to perform computations.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        tuple: A tuple containing trained model, train_losses, and train_accuracies.
    """
    model.to(device)
    train_loader = DataLoader(TensorDataset(train_images, train_masks), batch_size=3, shuffle=True)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    accuracy_function = BinaryAccuracy(threshold=0.5).to(device)

    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        # Set the model in training mode
        model.train()
        # Initialize the total training loss and training accuracy
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0

        # Loop over the training set
        for (x, y) in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            # Send the input to the device
            x, y = x.to(device), y.to(device)
            # Perform a forward pass and calculate the training loss
            pred = model(x)
            loss = custom_loss(pred, y, alpha=0.5)
            # Zero out any previously accumulated gradients, perform backpropagation, and update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Add the loss to the total training loss so far
            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy_function(torch.sigmoid(pred), y.int())

        # Calculate average loss and accuracy for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_accuracy = epoch_train_accuracy / len(train_loader)

        # Append to lists
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy.item())

        # Print epoch metrics
        print(f'Epoch {epoch+1}, loss: {avg_train_loss:.4f}, Binary Accuracy: {avg_train_accuracy:.4f}')

    # Save trained model
    torch.save(model.state_dict(), 'trained_model.pt')

    return model, train_losses, train_accuracies

def plot_loss_accuracy(train_losses, train_accuracies):
    """
    Plot the training loss and accuracy over epochs.

    Args:
        train_losses (list): List of training losses for each epoch.
        train_accuracies (list): List of training accuracies for each epoch.
    """
    epochs_idx = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_idx, train_losses, marker='o', color='b', linestyle='-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs_idx)
    plt.grid(True)
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_idx, train_accuracies, marker='o', color='r', linestyle='-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs_idx)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('Training_loss_acc.png')
    plt.show()

def dice_similarity_coefficient(pred, target, smooth=1e-5):
    """
    Calculate the Dice Similarity Coefficient.

    Args:
        pred (torch.Tensor): Predicted output.
        target (torch.Tensor): Ground truth.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Dice Similarity Coefficient.
    """
    pred = (pred > 0.5).float()  # Binarize predictions
    intersection = torch.sum(pred * target)
    dice_coeff = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return dice_coeff

def evaluate_model(model, test_images, test_masks, device):
    """
    Evaluate the model on the test set.

    Args:
        model (torch.nn.Module): Trained neural network model.
        test_images (torch.Tensor): Tensor containing test images.
        test_masks (torch.Tensor): Tensor containing corresponding test masks.
        device (torch.device): Device to perform computations.

    Returns:
        tuple: A tuple containing DSC list, accuracy list, lists of real, predicted, and ground truth images.
    """
    model.eval()  # Set model to evaluation mode
    test_loader = DataLoader(TensorDataset(test_images, test_masks), batch_size=1, shuffle=False)  # Batch size 1 for inference
    accuracy_function = BinaryAccuracy(threshold=0.5).to(device)
    real_list, pred_list, ground_list = [], [], []
    dsc_list, accuracy_list = [], []

    for images, masks in tqdm(test_loader, desc='Evaluating'):
        with torch.no_grad():
            # Move data to device
            images, masks = images.to(device), masks.to(device)

            # Make predictions
            preds = model(images)
            preds = torch.sigmoid(preds)

            # Compute DSC for each sample
            dsc = dice_similarity_coefficient(preds, masks, smooth=1e-5).cpu()
            accuracy = accuracy_function(preds, masks.int())
            dsc_list.append(dsc)
            accuracy_list.append(accuracy.item())
            # remove the batch dimension
            real_list.append(images.squeeze(0))
            pred_list.append(preds.squeeze(0))
            ground_list.append(masks.squeeze(0))

    return dsc_list, accuracy_list, real_list, pred_list, ground_list

def plot_dsc_accuracy_histogram(dsc_list, accuracy_list, name):
    """
    Plot histograms for Dice Similarity Coefficient and Test Accuracy.

    Args:
        dsc_list (list): List of Dice Similarity Coefficient values.
        accuracy_list (list): List of Test Accuracy values.
        name (str): Name for saving the histogram plot.
    """
    plt.figure(figsize=(10, 5))

    # Plot histogram for Dice Similarity Coefficient
    plt.subplot(1, 2, 1)
    plt.hist(dsc_list, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Dice Similarity Coefficient Histogram')
    plt.xlabel('Dice Similarity Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Plot histogram for Test Accuracy
    plt.subplot(1, 2, 2)
    plt.hist(accuracy_list, bins=20, color='salmon', edgecolor='black', alpha=0.7)
    plt.title('Accuracy Histogram')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{name}_dsc_accuracy_histogram.png')  # Corrected the saving filename

def find_examples(images, masks, predictions, dsc_values, num_examples=3):
    """
    Find examples with best, worst, and median Dice Similarity Coefficients.

    Args:
        images (list): List of real images.
        masks (list): List of ground truth masks.
        predictions (list): List of predicted masks.
        dsc_values (list): List of Dice Similarity Coefficient values.
        num_examples (int): Number of examples to find for each category.

    Returns:
        dict: Dictionary containing examples categorized as 'best', 'worst', and 'median'.
    """
    dsc_values = np.array(dsc_values)  # Convert to NumPy array
    best_indices = np.argsort(dsc_values)[-num_examples:][::-1]
    worst_indices = np.argsort(dsc_values)[:num_examples]
    median_indices = np.argsort(np.abs(dsc_values - 0.5))[:num_examples]

    examples = {
        'best': [],
        'worst': [],
        'median': []
    }

    for index in best_indices:
        examples['best'].append((images[index], masks[index], predictions[index]))
    for index in worst_indices:
        examples['worst'].append((images[index], masks[index], predictions[index]))
    for index in median_indices:
        examples['median'].append((images[index], masks[index], predictions[index]))

    return examples

def plot_examples(examples, save_path='example_plots/'):
    """
    Plot example images with real, ground truth, and predicted masks.

    Args:
        examples (dict): Dictionary containing examples categorized as 'best', 'worst', and 'median'.
        save_path (str): Path to save the plots.
    """
    import os

    categories = ['best', 'worst', 'median']

    # Create a directory to save the plots if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    for category in categories:
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        fig.suptitle(category.capitalize(), fontsize=16)  # Add category name to the main title

        for i, (image, mask, prediction) in enumerate(examples[category]):
            # Move tensors to CPU before converting to NumPy arrays
            image_np = image.squeeze().cpu().numpy()  # Squeeze singleton dimension
            mask_np = mask.squeeze().cpu().numpy()  # Squeeze singleton dimension
            prediction_np = prediction.squeeze().cpu().numpy()  # Squeeze singleton dimension

            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title('Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(prediction_np, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{category}_examples.png'))  # Save the plot
        plt.close(fig)  # Close the figure to release memory


if __name__ == '__main__':
    # Prepare training dataset and testing dataset
    images_dataset_dir = "Dataset/Images"
    images_tensor = prepare_images_tensor(images_dataset_dir)
    masks_dataset_dir = "Dataset/Segmentations"
    masks_tensor = prepare_masks_tensor(masks_dataset_dir)
    train_images, train_masks, test_images, test_masks = split_dataset(images_tensor, masks_tensor)
    
    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define and train the model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model, train_losses, train_accuracies = training(model, train_images, train_masks, device, epochs=10, learning_rate=0.1)
    
    # Plot training loss and accuracy
    plot_loss_accuracy(train_losses, train_accuracies)
    
    # Load the trained model
    model.load_state_dict(torch.load("trained_model.pt"))
    
    # Evaluate the model on the test set
    print('Reading completed.\nPredicting testing images using pre-trained model trained_model.pt')
    dsc_test, accuracy_test, real_test, pred_test, ground_test = evaluate_model(model, test_images, test_masks, device)
    
    # Plot histograms for test set
    plot_dsc_accuracy_histogram(dsc_test, accuracy_test, name="test")
    
    # Find and plot examples for test set
    examples = find_examples(real_test, ground_test, pred_test, dsc_test, num_examples=3)
    plot_examples(examples, save_path='example_plots_test/')
    
    # Evaluate the model on the training set
    dsc_train, accuracy_train, real_train, pred_train, ground_train = evaluate_model(model, train_images, train_masks, device)
    
    # Plot histograms for training set
    plot_dsc_accuracy_histogram(dsc_train, accuracy_train, name="train")
    
    # Find and plot examples for training set
    examples = find_examples(real_train, ground_train, pred_train, dsc_train, num_examples=3)
    plot_examples(examples, save_path='example_plots_train/')
