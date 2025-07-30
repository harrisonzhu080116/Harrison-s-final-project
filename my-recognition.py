#!/usr/bin/python3

import jetson_inference

import jetson_utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("filename", type=str, help="filename of the image to process")

parser.add_argument("--model", type=str, default="/home/harrison/jetson-inference/python/training/classification/models/plants-crops/resnet18.onnx", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
parser.add_argument("--labels", type=str, default="/home/harrison/jetson-inference/python/training/classification/models/plants-crops/labels.txt", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

input_blob = "input_0"
output_blob = "output_0"

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(model=opt.model, labels=opt.labels, input_blob=input_blob, output_blob=output_blob)

class_idx, confidence = net.Classify(img)
class_description = net.GetClassDesc(class_idx)

print("The image is classified as a", class_description, "at class a" , str(class_idx)," with a confidence of",str(confidence*100)+"%")





