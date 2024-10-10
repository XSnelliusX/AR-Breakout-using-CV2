import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Ball properties
ball_radius = 20
ball_color = (0, 255, 0)  # Green color

# Game state
game_over = False
is_winner = False

# Box properties
boxes_initialized = False
box_color = (255, 0, 0)  # Red color
box_size = (80, 40)  # Width, Height
gap = 5  # Gap between boxes
boxes = []

def initialize_boxes(image_width, box_size, gap, max_rows=3):
    boxes = []
    box_width, box_height = box_size
    
    # Determine the number of boxes per row based on image width
    num_boxes_per_row = image_width // (box_width + gap)
    
    # Calculate the number of boxes based on available vertical space
    num_boxes = int(num_boxes_per_row * max_rows)

    for i in range(num_boxes):
        x = (i % num_boxes_per_row) * (box_width + gap)
        y = gap + (i // num_boxes_per_row) * (box_height + gap)
        boxes.append([x, y, box_width, box_height])  # [x, y, w, h]
    
    return boxes

def game_over_screen(image):
    # Fill the screen with a black rectangle to display the game over text
    cv2.rectangle(image, (0, 0), (image_width, image_height), (0, 0, 0), -1)

    # Text parameters
    game_over_text = 'Game Over'
    restart_text = 'Press R to Restart or ESC to Quit'

    # Get the text sizes
    (game_over_text_size, _) = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    (restart_text_size, _) = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Calculate the center positions for the text
    game_over_x = (image_width - game_over_text_size[0]) // 2
    game_over_y = (image_height + game_over_text_size[1]) // 2  # Vertical center

    restart_x = (image_width - restart_text_size[0]) // 2
    restart_y = game_over_y + 60  # Slightly below the first text line

    # Draw the texts centered
    cv2.putText(image, game_over_text, (game_over_x, game_over_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(image, restart_text, (restart_x, restart_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
def win_screen(image):
    # Fill the screen with a black rectangle to display the win text
    cv2.rectangle(image, (0, 0), (image_width, image_height), (0, 0, 0), -1)

    # Text parameters
    win_text = 'You Win!'
    restart_text = 'Press R to Restart or ESC to Quit'

    # Get the text sizes
    (win_text_size, _) = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    (restart_text_size, _) = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Calculate the center positions for the text
    win_x = (image_width - win_text_size[0]) // 2
    win_y = (image_height + win_text_size[1]) // 2  # Vertical center

    restart_x = (image_width - restart_text_size[0]) // 2
    restart_y = win_y + 60  # Slightly below the first text line

    # Draw the texts centered
    cv2.putText(image, win_text, (win_x, win_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(image, restart_text, (restart_x, restart_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            
def wall_collision(window_width, window_height, ball_position, ball_velocity):
    global game_over

    # Bounce off the left wall
    if ball_position[0] - ball_radius < 0:
        ball_position[0] = ball_radius
        ball_velocity[0] = -ball_velocity[0]

    # Bounce off the right wall
    if ball_position[0] + ball_radius > window_width:
        ball_position[0] = window_width - ball_radius
        ball_velocity[0] = -ball_velocity[0]

    # Bounce off the top wall
    if ball_position[1] - ball_radius < 0:
        ball_position[1] = ball_radius
        ball_velocity[1] = -ball_velocity[1]

    # Check for game over (touching the bottom wall)
    if ball_position[1] + ball_radius > window_height:
        game_over = True  # Set game over flag
        return ball_position, [0, 0]  # Stop the ball

    return ball_position, ball_velocity

def line_collision(line_start, line_end, ball_position, ball_velocity):
    # Vector from line start to line end
    line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
    # Vector from line start to ball position
    ball_vec = np.array([ball_position[0] - line_start[0], ball_position[1] - line_start[1]])

    # Squared length of the line segment
    line_length_sq = np.dot(line_vec, line_vec)

    # Projection factor of the ball vector onto the line vector
    t = np.dot(ball_vec, line_vec) / line_length_sq
    # Clamp t to the range [0, 1] to ensure the closest point is on the line segment
    t = np.clip(t, 0, 1)

    # Closest point on the line segment to the ball
    closest_point = np.array(line_start) + t * line_vec
    # Distance from the ball to the closest point on the line segment
    dist_to_closest_point = np.linalg.norm(np.array(ball_position) - closest_point)

    # If the ball is touching or intersecting the line segment (within the ball radius + 5 pixels)
    if dist_to_closest_point <= ball_radius + 5:
        # Normal vector to the line segment
        normal_vec = np.array([-line_vec[1], line_vec[0]]) / np.sqrt(line_length_sq)
        # Reflect the ball's velocity over the normal vector
        ball_velocity = np.array(ball_velocity, dtype=np.float64)
        ball_velocity -= 2 * np.dot(ball_velocity, normal_vec) * normal_vec

    return ball_velocity

def box_collision(boxes, ball_position, ball_velocity):
    for box in boxes:
        x, y, w, h = box
        ball_x, ball_y = ball_position

        # Check for collision with the left or right edges
        if y < ball_y < y + h:  # Ball's y-coordinate is within the box's y-range
            if abs(ball_x - x) <= ball_radius and ball_velocity[0] > 0:  # Left edge collision
                ball_velocity[0] = -abs(ball_velocity[0])  # Bounce to the left
                boxes.remove(box)
                break
            elif abs(ball_x - (x + w)) <= ball_radius and ball_velocity[0] < 0:  # Right edge collision
                ball_velocity[0] = abs(ball_velocity[0])  # Bounce to the right
                boxes.remove(box)
                break

        # Check for collision with the top or bottom edges
        if x < ball_x < x + w:  # Ball's x-coordinate is within the box's x-range
            if abs(ball_y - y) <= ball_radius and ball_velocity[1] > 0:  # Top edge collision
                ball_velocity[1] = -abs(ball_velocity[1])  # Bounce upward
                boxes.remove(box)
                break
            elif abs(ball_y - (y + h)) <= ball_radius and ball_velocity[1] < 0:  # Bottom edge collision
                ball_velocity[1] = abs(ball_velocity[1])  # Bounce downward
                boxes.remove(box)
                break

    return boxes, ball_velocity

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Initialize boxes and ball position/velocity (only once)
        if not boxes_initialized:
            image_height, image_width, _ = image.shape
            ball_position = [np.random.randint(ball_radius + 50, image_width - ball_radius - 50), image_height - 50] # Initial position (x, y)
            ball_velocity = [np.random.randint(-15, 15), -15] # Initial velocity (x, y)
            boxes = initialize_boxes(image_width, box_size, gap, 1)
            boxes_initialized = True
        
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        # Check for win condition
        if not game_over and not boxes:
            game_over = True  # Reusing the game_over flag to also indicate a win state
            is_winner = True  # Indicate that the game is won
        
        # Show win screen if the player has won
        if game_over and is_winner:
            win_screen(image)
        # Show game over screen if the player lost
        elif game_over:
            game_over_screen(image)
            
        # Game logic if the game is not over
        if not game_over:
            # Update ball position
            ball_position[0] += ball_velocity[0]  # Update x position
            ball_position[1] += ball_velocity[1]  # Update y position

            # Wall Collision detection
            ball_position, ball_velocity = wall_collision(image_width, image_height, ball_position, ball_velocity)

            # Process the hand detection
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                    index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                    thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
                    thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    cv2.line(image, (int(index_x), int(index_y)), (int(thumb_x), int(thumb_y)), (0, 255, 0), 2)

                    if not game_over:
                        ball_velocity = line_collision((int(index_x), int(index_y)), (int(thumb_x), int(thumb_y)), ball_position, ball_velocity)

            # Draw the ball
            cv2.circle(image, (int(ball_position[0]), int(ball_position[1])), ball_radius, ball_color, -1)

            # Draw the boxes and handle collisions
            for box in boxes[:]:  # Iterate over a copy of the list
                box_x, box_y, box_w, box_h = box
                # Draw the box
                cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 0, 0), -1)

                # Check for collision with the ball
                if (box_x < ball_position[0] < box_x + box_w and
                        box_y < ball_position[1] < box_y + box_h):
                    # Remove the box
                    boxes.remove(box)
                    # Bounce the ball
                    ball_velocity[1] = -ball_velocity[1]

        # Display the frame
        cv2.imshow('AR Breackout', image)

        # Check for key presses
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # 'r' key to restart
            ball_position = [np.random.randint(ball_radius + 50, image_width - ball_radius - 50), image_height - 50]
            ball_velocity = [np.random.randint(-15, 15), -15]
            boxes_initialized = False
            game_over = False
            is_winner = False  # Reset winning state

    cap.release()
