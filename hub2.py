import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import mediapipe as mp


# -----------------------------
# 손가락 개수 추정 함수
# -----------------------------
def count_fingers_mediapipe(img):
    img = cv2.resize(img, (640, 480))
    output = img.copy()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # MediaPipe는 RGB 사용
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,      # 사진 한 장 분석
        max_num_hands=1,             # 손 하나만 감지
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            return output, 0, "손을 찾을 수 없습니다."

        hand_landmarks = result.multi_hand_landmarks[0]

        # 랜드마크 그리기
        mp_draw.draw_landmarks(
            output,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        h, w, _ = img.shape

        # 랜드마크를 픽셀 좌표로 변환
        lm = []
        for point in hand_landmarks.landmark:
            lm.append((int(point.x * w), int(point.y * h)))

        fingers = []

        # -----------------------------
        # 엄지 판정
        # -----------------------------
        # 오른손/왼손에 따라 엄지 방향이 다름
        handedness = result.multi_handedness[0].classification[0].label
        # label: "Left" 또는 "Right"

        thumb_tip = lm[4]
        thumb_ip = lm[3]

        if handedness == "Right":
            if thumb_tip[0] < thumb_ip[0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if thumb_tip[0] > thumb_ip[0]:
                fingers.append(1)
            else:
                fingers.append(0)

        # -----------------------------
        # 나머지 손가락 판정
        # tip이 pip보다 위에 있으면 펴진 것으로 판단
        # y좌표는 위로 갈수록 작음
        # -----------------------------
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, finger_pips):
            if lm[tip][1] < lm[pip][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        count = sum(fingers)

        cv2.putText(
            output,
            f"Fingers: {count}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3
        )

        cv2.putText(
            output,
            f"Hand: {handedness}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2
        )

        return output, count, "성공"


# -----------------------------
# 파일 선택
# -----------------------------
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="손 사진을 선택하세요",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("All files", "*.*")
    ]
)

if not file_path:
    print("사진을 선택하지 않았습니다.")
    exit()


# -----------------------------
# 이미지 읽기
# 한글 경로 대응
# -----------------------------
img_array = np.fromfile(file_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
git
if img is None:
    print("이미지를 읽을 수 없습니다.")
    print("선택한 파일 경로:", file_path)
    exit()


# -----------------------------
# 손가락 개수 세기
# -----------------------------
result_img, count, message = count_fingers_mediapipe(img)

print(message)
print("추정 손가락 개수:", count)


# -----------------------------
# 화면 출력
# -----------------------------
cv2.imshow("MediaPipe Result", result_img)

print("창에서 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()