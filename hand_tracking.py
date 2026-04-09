import cv2
import mediapipe as mp


class HandTracker:
    """Detects and tracks hands in video frames using MediaPipe."""

    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._results = None

    def find_hands(self, frame, draw=True):
        """Process a BGR frame, detect hands, and optionally draw landmarks.

        Results are cached so that :meth:`get_landmark_positions` can reuse
        them without reprocessing the same frame.

        Args:
            frame: BGR image from OpenCV.
            draw: Whether to draw landmarks and connections on the frame.

        Returns:
            The (possibly annotated) frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._results = self.hands.process(rgb_frame)

        if self._results.multi_hand_landmarks:
            for hand_landmarks in self._results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

        return frame

    def get_landmark_positions(self, frame, hand_index=0):
        """Return pixel coordinates of each landmark for the specified hand.

        Uses cached results from the last :meth:`find_hands` call when
        available, avoiding redundant MediaPipe processing.

        Args:
            frame: BGR image (used only for its dimensions).
            hand_index: Index of the hand whose landmarks to return.

        Returns:
            List of (id, x, y) tuples, or an empty list if no hand is found.
        """
        if self._results is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._results = self.hands.process(rgb_frame)

        landmark_list = []
        if self._results.multi_hand_landmarks:
            if hand_index < len(self._results.multi_hand_landmarks):
                hand = self._results.multi_hand_landmarks[hand_index]
                h, w, _ = frame.shape
                for lm_id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append((lm_id, cx, cy))

        return landmark_list
