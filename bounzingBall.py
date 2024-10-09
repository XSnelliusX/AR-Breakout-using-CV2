import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Ball properties
ball_radius = 100
ball_color = (0, 255, 0)  # Green color
initial_ball_position = [400, 200]  # Initial position (x, y)
ball_position = initial_ball_position.copy()
ball_velocity = [12.5, -7.5]    # Initial velocity (vx, vy)

def wall_collision(window_width, window_height, ball_position, ball_velocity, ball_color):
    bounce = False
    # Bounce off the left wall
    if ball_position[0] - ball_radius < 0:
        ball_position[0] = ball_radius
        ball_velocity[0] = -ball_velocity[0]
        bounce = True

    # Bounce off the right wall
    if ball_position[0] + ball_radius > window_width:
        ball_position[0] = window_width - ball_radius
        ball_velocity[0] = -ball_velocity[0]
        bounce = True

    # Bounce off the top wall
    if ball_position[1] - ball_radius < 0:
        ball_position[1] = ball_radius
        ball_velocity[1] = -ball_velocity[1]
        bounce = True

    # Bounce off the bottom wall
    if ball_position[1] + ball_radius > window_height:
        ball_position[1] = window_height - ball_radius
        ball_velocity[1] = -ball_velocity[1]
        bounce = True
    
    if bounce:
        ball_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    return ball_position, ball_velocity, ball_color


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    image_height, image_width, _ = image.shape
    
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)

    # Update ball position
    ball_position[0] += ball_velocity[0]  # Update x position
    ball_position[1] += ball_velocity[1]  # Update y position

    # Collision detection
    ball_position, ball_velocity, ball_color = wall_collision(image_width, image_height, ball_position, ball_velocity, ball_color)

    cv2.circle(image, (int(ball_position[0]), int(ball_position[1])), ball_radius, ball_color, -1)
    cv2.imshow('Bouncing Ball', image)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    
cap.release()
