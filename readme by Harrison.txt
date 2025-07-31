Harrison's Final Project

The script loads an image file, runs it through an AI model (like GoogLeNet or ResNet-18), and prints the most likely object in the image with a confidence
score.


# 🧠 Jetson Image Classification

This project is a simple image classification tool using NVIDIA Jetson's `jetson-inference` framework. It allows you to classify a single image using a pre-trained deep learning model such as GoogLeNet or ResNet-18.

## 📸 What It Does

Given an input image, the script:
1. Loads the image.
2. Loads a neural network model.
3. Classifies the image.
4. Prints out the top predicted class and its confidence.

---

## 🚀 Requirements

- NVIDIA Jetson device (e.g., Jetson Nano, Xavier, Orin)
- [jetson-inference library](https://github.com/dusty-nv/jetson-inference)
- Python 3

To install jetson-inference:
```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make
sudo make install


Usage
python3 classify_image.py <image_filename> [--network <model_name>]

For example
python3 classify_image.py dog.jpg --network=resnet-18

Parameters:
<image_filename>: Path to the image you want to classify.

--network: (Optional) The name of the pre-trained model to use. Default is googlenet.

Supported models include:

googlenet (default)

resnet-18

alexnet

vgg-16

and more...

See all available models:
imagenet --help





🧠 Example Output

The image is classified as: Golden Retriever (class ID: 212) with confidence: 89.45%


📁 File Structure

project/
├── classify_image.py      # Main Python script
├── dog.jpg                # Example image (optional)
└── README.md              # This file

🧩 Notes

The script uses models pre-trained on ImageNet (1000-class object recognition).

Jetson hardware with GPU acceleration is required for best performance.

Make sure your image is accessible and supported by jetson_utils.loadImage() (JPG, PNG, etc.).

📜 License

This project is based on NVIDIA Jetson Inference, which is released under the MIT License. This script is free to use and modify for educational or research purposes.






About the code, here's how it works.

The python file is like:"





#!/usr/bin/python3

import jetson_inference
import jetson_util
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)


class_idx, confidence = net.Classify(img)
class_description = net.GetClassDesc(class_idx)

print("The image is classified as a", class_description, "at class a" , str(class_idx)," with a confidence of",str(confidence*100)+"%")


"

How does it work:

1. Import Libraries

jetson_inference: Contains pre-trained deep learning models like imageNet, detectNet, etc.

jetson_utils: Provides functions for working with images, video, and CUDA.

argparse: Handles command-line arguments (like image filename and model type).



2. Parse Command-Line Arguments

filename: The name of the image file to classify (required).

--network: Optional. The name of the model to use (default is googlenet).



3. Load the Image

This loads the image (e.g., dog.jpg) into GPU memory, ready for processing.



4. Load the Neural Network

This loads the chosen model (e.g., googlenet, resnet-18) for image classification.

These models are already trained on ImageNet, which recognizes 1000+ everyday objects (like cat, car, tree, airplane, etc.).



5. Run the Model (Classify the Image)

Classify(img) returns:

class_idx: ID of the predicted class (e.g., 212 might be "Golden Retriever").

confidence: A value between 0 and 1 showing how sure the model is.



6. Get Class Description

Converts class ID (like 212) to a human-readable label (like "Golden Retriever").



7. Print the Result






