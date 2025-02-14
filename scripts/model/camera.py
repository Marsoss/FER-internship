import cv2
# sourcery skip: dont-import-test-modules
from test_model_torch import CNNTester, choose_filename
import argparse

def parse(str:str, char:str):
    return str.split(char)

def take_picture(model_name='models/unet1_FER2013.pth'):
    # Create a VideoCapture object to capture video from the default camera
    cap = cv2.VideoCapture(0)
    if model_name is None:
        file_filter = [("PTH Files", "*.pth")]
        model_name = choose_filename(file_filter, start_folder='models')
        print(f"Model name: {model_name}")
    dataset_name = parse(parse(parse(model_name, '/')[1], '.')[0], '_')[1]
    tester = CNNTester(model_name, dataset_path=f"../datasets/{dataset_name}",model_color_mode='grayscale')

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
    print("Press 'Space' to capture an image or 'q' to quit")

    while cap.isOpened():

        flipped_frame = cv2.flip(frame, 1)
        # Show the captured frame in a window    
        cv2.imshow("Camera", flipped_frame)

        # Check if the window has been closed
        if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user")
            break

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # Capture image when space bar is pressed
        if key == 32:  # Space bar keycode
            # Save the frame as an image

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            eps = 0
            for (x, y, w, h) in faces:
                cropped_face = frame[y-eps:y+h+eps, x-eps:x+w+eps]

            cv2.imwrite("captured_image.jpg", cropped_face)

            print("Image captured successfully!")
            tester.test(image_name='captured_image.jpg')
            print("Press 'Space' to capture an image or 'q' to quit")



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture an image using the webcam and test it with a CNN model.")
    parser.add_argument('--model_name', type=str, default='models/unet1_FER2013.pth', help='Path to the model file')
    args = parser.parse_args()

    # Call the function to take a picture
    take_picture(model_name=args.model_name)
