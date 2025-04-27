# NVIDIA-Jetson-Nano-Path-Following

![jetbot](https://github.com/user-attachments/assets/81ceaedf-59ff-4aa7-a4b9-ea1faf85d98d)

## Introduction
This project demonstrates how to implement autonomous path following on a JetBot. By training various models on path images, the JetBot can learn to navigate itself along the path using real-time camera input.

## Methodology

### Data Collection
The project uses a dataset of images collected from the JetBot's camera while navigating. Each image is labeled with (x, y) coordinates representing the target direction. The dataset is organized in a way that filenames contain the steering information (extracted using the `get_x` and `get_y` functions).

Two methods for data collection are available:

- **`data_collection_gamepad.ipynb`**: This notebook uses a gamepad controller to label images with x and y values. The user places a 'green dot' on the image to indicate the target direction.
- **`data_collection.ipynb`**: This notebook uses a clickable image widget to annotate images. Clicking on the live image feed saves the image with the clicked coordinates.

Rationale:

Using (x, y) coordinates provides a continuous representation of the steering direction, enabling finer control.
Storing coordinates in filenames simplifies data loading and labeling.

### Dataset Preparation
The dataset is loaded using a custom `XYDataset` class that inherits from PyTorch's Dataset. Data augmentation is applied, including:

- Random horizontal flips (optional) - It increases data variability, especially useful if the path is symmetrical.
- Color jittering - It makes the model invariant to slight changes in lighting conditions.
- Resizing to 224x224 pixels - Standardizes input size for the model.
- Normalization using ImageNet statistics - It helps the model converge faster by scaling the input to a suitable range.

The dataset is split into training (90%) and testing (10%) sets. As a separate test set is needed for evaluating the model's performance and preventing overfitting.

### Model Training

Multiple models were evaluated for the path following task:

-   **ResNet18**: Trained for 70 epochs - Considered this as a baseline model, as presented in the original notebook examples. ResNet18 provides a good balance between representational capacity and computational cost. We wanted to assess its performance as a starting point.

-   **MobileNetV2**: Trained for 13 epochs - We used this model to explore a more efficient architecture. MobileNetV2 is designed for mobile devices and is known for its low computational footprint, which is crucial for real-time performance on the Jetson Nano. We aimed to see if we could maintain acceptable accuracy while significantly improving speed.

-   **ShuffleNet**: Trained for 18 epochs - We also wanted to test lightweight models, specifically ShuffleNetV2, to push the limits of efficiency. ShuffleNetV2 is another architecture optimized for mobile devices, utilizing channel shuffling to reduce computation. Our goal was to determine the minimum model size that still allowed for effective path following.

The epochs were determined empirically, by monitoring the validation loss during training, to prevent overfitting. We stopped training when the validation loss started to plateau or increase, indicating that the model was beginning to memorize the training data rather than generalize to new data.

Training details:
- Loss function: Mean Squared Error (MSE) - An appropriate function for regression tasks, measuring the difference between predicted and actual (x, y) coordinates.
- Optimizer: Adam
- Batch size: 8

### Model Optimization
For real-time performance, the trained PyTorch models were optimized using TensorRT:

```python
from torch2trt import torch2trt

# Convert model to TensorRT format
model_trt = torch2trt(model, [dummy_input], fp16_mode=True) # From live_demo_build_trt.ipynb
```

## Implementation
The live demonstration uses:

- Camera input preprocessing with normalization
- Real-time inference with the TensorRT-optimized model
- PD (Proportional-Derivative) control for smooth steering
- Interactive sliders to tune parameters:
  - Speed gain (It controls the overall speed of the JetBot)
  - Steering gain (It adjusts the responsoveness of steering)
  - Steering derivative gain (kd) (It provides stability)
  - Steering bias (It compensated for any inherent steering bias.)

### Key Components

#### Custom Models
The project includes various neural network models:
- ResNet18
- MobileNetV2
- ShuffleNet

#### Real-time Control Logic

```python
def execute(change):
    image = change['new']
    xy = model_trt(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

    angle = np.arctan2(x, y)
    pid = angle * steering_gain_slider.value + (angle - angle_last) * steering_dgain_slider.value

    robot.left_motor.value = max(min(speed_slider.value + steering_slider.value, 1.0), 0.0)
    robot.right_motor.value = max(min(speed_slider.value - steering_slider.value, 1.0), 0.0)
```

## Results
The system successfully enables the JetBot to follow paths smoothly using deep learning and computer vision. The TensorRT optimization allows real-time inference even on the resource-constrained Jetson Nano hardware.

## Performance:

The TensorRT optimization significantly improved inference speed, enabling real-time control.
The PD control method provided smooth and stable steering.

## Challenges:

Collecting a diverse and representative dataset was crucial for good performance.
Balancing model complexity and inference speed was important for real-time operation.
Tuning the control parameters required careful experimentation.

## Future Improvements
- Collect more diverse training data for better generalization
- Add collision avoidance capabilities
- Implement adaptive speed control based on path complexity

---
## Instructions to run project
To get started with this project, follow these steps:

- Set up your JetBot:

Ensure your JetBot is assembled correctly and the camera is properly connected.
Install the required software and dependencies on your Jetson Nano, including PyTorch and the JetBot libraries.
Collect Data:

- Use either data_collection.ipynb or data_collection_gamepad.ipynb to collect images of the road or path you want the JetBot to follow. Label the images as described in the notebook. Save the collected data.

- Transfer Data (if necessary):

If you collected data on the JetBot, you may need to transfer it to a more powerful machine for training. The notebooks provide instructions on how to zip the dataset for easier transfer.

- Train the Model:

Run the train_model-model_name.ipynb notebook to train a model on your collected data. Monitor the training process and adjust hyperparameters as needed. Save the best performing model.

- Optimize the Model (Optional but Recommended):

Use the live_demo_build_trt.ipynb notebook to optimize the trained model with TensorRT for better performance on the Jetson Nano.

- Run the Live Demo:

Execute the live_demo_trt.ipynb notebook to see the JetBot autonomously follow the path.
Use the provided sliders to fine-tune the JetBot's behavior.
Troubleshooting:

* If the JetBot is not performing as expected, consider collecting more data, adjusting training parameters, or fine-tuning the control parameters in the live demo.
