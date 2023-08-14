import cv2
def detect_faces(camera_index=0):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the specified camera index
    cap = cv2.VideoCapture(0)
    

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Convert the frame to grayscale (face detection works on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (229, 225, 12), 4)

        # Display the frame with face rectangles
        cv2.imshow('Face Detection', frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('Q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 0 with the appropriate camera index if using an external camera
  detect_faces(camera_index=1)
