import cv2
# sourcery skip: dont-import-test-modules
from test_model_torch import CNNTester

def take_picture():
    # Create a VideoCapture object to capture video from the default camera
    cap = cv2.VideoCapture(0)

    tester = CNNTester('models/unet1_best_10.pth', 96, 96, 'rgb')
    tester.load_model()
    print("model loaded")
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Failed to open camera")
        return

    cv2.namedWindow("Camera")

    face_cascade = cv2.CascadeClassifier(
        f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml'
    )


    # Read a frame from the video stream
    ret, frame = cap.read()

    while True:

        flipped_frame = cv2.flip(frame, 1)
        # Show the captured frame in a window    
        cv2.imshow("Camera", flipped_frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # Capture image when space bar is pressed
        if key == 32:  # Space bar keycode
            # Save the frame as an image

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cropped_face = frame[y:y+h, x:x+w]

            cv2.imwrite("captured_image.jpg", cropped_face)

            print("Image captured successfully!")
            tester.test(image_name='captured_image.jpg')


        # Break the loop if 'q' or 'Esc' key is pressed
        if key in [ord('q'), 27]:
            print("Image capture cancelled.")
            break

        # Read the next frame
        ret, frame = cap.read()

    # Release the VideoCapture object and close the camera
    cap.release()
    # Close the window
    cv2.destroyAllWindows()

    
# Call the function to take a picture
take_picture()
