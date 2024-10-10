# AR Breakout Game

This repository contains an very basic Augmented Reality (AR) version of the classic game **Breakout**. In this version, the traditional static platform at the bottom of the screen is replaced by a dynamic, interactive platform—**your hand**! Specifically, a line is drawn between your thumb and index finger, allowing you to control the paddle in real-time.

## Features

- **Hand-controlled paddle**: Instead of using a keyboard or mouse, the paddle is controlled by tracking your hand movements. A virtual line between your thumb and index finger acts as the platform to bounce the ball.
- **Augmented Reality (AR)**: Using computer vision and hand tracking, the game leverages your camera feed to create an interactive AR experience.
- **Classic Breakout gameplay**: Break the bricks by bouncing the ball off your hand-controlled paddle while avoiding the ball falling off the screen.

## How it Works

- The game uses the **MediaPipe Hands** solution to detect and track the position of your hand and OpenCV for the rest like drawing the Ball.
- A line is drawn between the tip of your thumb and index finger, which serves as the paddle in the game.
- You move your hand in front of the camera to control the paddle, bouncing the ball to break bricks.

## Requirements

To run the game, you'll need the following:

- **Python 3.7+**
- **OpenCV** for real-time computer vision
- **MediaPipe** for hand tracking

You can install the required packages using the following command:

```bash
pip install opencv-python mediapipe
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/XSnelliusX/AR-Breakout-using-CV2.git
cd AR-Breakout-using-CV2
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the game:

```bash
python ar_breakout.py
```

Make sure your webcam is enabled, as the game uses real-time video feed to track your hand movements.



## Gameplay Controls

- **Move your hand**: Place your hand in front of the camera, and adjust the position of your thumb and index finger to control the paddle. 
- **Avoid missing the ball**: Just like in the original game, prevent the ball from falling off the screen by bouncing it with your virtual paddle.
- **Break all bricks**: The goal is to break all the bricks on the screen using the ball.

## Test Files
This repository also includes two test scripts that were used for experimenting with MediaPipe and collision detection. These files are not required to play the game but are helpful for understanding how the individual components work:

- `bounzingBall.py`:
This script tests a simple bouncing ball animation with boundary collision detection. It showcases how a ball can interact with the edges of the screen, changing direction and color upon impact.

- `handRecognizion.py`:
This script demonstrates how to use MediaPipe to detect and track hand landmarks in real-time, focusing on the thumb and index finger. It draws the detected hand landmarks on the webcam feed for visual feedback.

These files were developed as experiments to test functionality and do not directly contribute to the game.
