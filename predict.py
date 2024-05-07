import numpy as np
import pandas as pd
import cv2
import os
import turtle
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(40,40,3), include_top=False, weights='imagenet', pooling='avg')
pretrained_model3.trainable = False
inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)
outputs3 = tf.keras.layers.Dense(4, activation='softmax')(x3)
model1 = tf.keras.Model(inputs=inputs3, outputs=outputs3)

checkpoint_path = "training_1/cp.weights.h5"

train_dir = './data/'
labels_dict = {0: 'UP', 1: 'DOWN', 3: 'RIGHT', 2: 'LEFT'}

BUFFER = 20

Name = []
for file in os.listdir(train_dir):
    Name += [file]
print(Name)
print(len(Name))

model1.load_weights(checkpoint_path)

screen = turtle.Screen()
screen.setup(width=600, height=600)
screen.bgcolor("white")
screen.title("Turtle Control")

# Create turtle
t = turtle.Turtle()
t.shape("turtle")
t.color("blue")
t.speed(0)

# Initialize list of last 5 directions
last_5_directions = []

def control_turtle(move):
    global last_5_directions
    
    # Add the current direction to the list
    last_5_directions.append(move)
    
    # Keep only the last 5 directions
    if len(last_5_directions) > BUFFER:
        last_5_directions = last_5_directions[-BUFFER:]
    
    # Check if all the last 5 directions are the same
    if len(set(last_5_directions)) == 1:
        # Move the turtle in the direction of the last 5 directions
        if move == 'UP':
            t.setheading(90)
            t.forward(1)
        elif move == 'DOWN':
            t.setheading(270)
            t.forward(1)
        elif move == 'LEFT':
            t.setheading(180)
            t.forward(1)
        elif move == 'RIGHT':
            t.setheading(0)
            t.forward(1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to the input size expected by the model
    frame_rgb = cv2.resize(frame, (40, 40))
    frame_rgb = frame_rgb / 255.0  # Normalize pixel values
    
    # Run predictions
    predictions = model1.predict(np.expand_dims(frame_rgb, axis=0))
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    
    # Get the predicted class label
    predicted_class_label = labels_dict[predicted_class_index]
    
    # Overlay the classification text on the frame
    cv2.putText(frame, f'{Name[predicted_class_index]}:{predicted_class_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame with the classification
    cv2.imshow('Video Feed', frame)

    control_turtle(predicted_class_label)
    
    # Check for key press to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
