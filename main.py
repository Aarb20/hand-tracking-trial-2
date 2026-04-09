import cv2
from hand_tracking import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tracker = HandTracker(max_hands=2, detection_confidence=0.5, tracking_confidence=0.5)

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break

        frame = tracker.find_hands(frame)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
