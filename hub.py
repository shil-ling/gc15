import cv2
import tkinter as tk
from tkinter import filedialog
import math
import numpy as np


# -----------------------------
# 손가락 개수 추정 함수
# -----------------------------
def count_fingers(img):
    # 이미지 크기 줄이기
    img = cv2.resize(img, (640, 480))

    # 원본 복사
    output = img.copy()

    # BGR → HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 피부색 범위 설정
    # 조명에 따라 이 값은 조금씩 바꿔야 할 수 있음
    lower_skin = (0, 30, 60)
    upper_skin = (25, 180, 255)

    # 피부색 영역만 마스크로 추출
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 노이즈 제거
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return output, mask, 0, "손을 찾을 수 없습니다."

    # 가장 큰 윤곽선을 손이라고 가정
    hand_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(hand_contour)

    if area < 3000:
        return output, mask, 0, "손 영역이 너무 작습니다."

    # 손 윤곽선 그리기
    cv2.drawContours(output, [hand_contour], -1, (0, 255, 0), 2)

    # 볼록 껍질 계산
    hull = cv2.convexHull(hand_contour, returnPoints=False)

    if hull is None or len(hull) < 3:
        return output, mask, 0, "손 모양을 분석할 수 없습니다."

    # 볼록 결함 계산
    defects = cv2.convexityDefects(hand_contour, hull)

    if defects is None:
        return output, mask, 0, "손가락 사이를 찾을 수 없습니다."

    finger_gaps = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]

        start = tuple(hand_contour[s][0])
        end = tuple(hand_contour[e][0])
        far = tuple(hand_contour[f][0])

        # 세 점 사이 거리 계산
        a = math.dist(start, end)
        b = math.dist(start, far)
        c = math.dist(end, far)

        # 코사인 법칙으로 각도 계산
        if b * c == 0:
            continue

        angle = math.degrees(
            math.acos((b * b + c * c - a * a) / (2 * b * c))
        )

        # 손가락 사이 골짜기 조건
        if angle < 90 and d > 10000:
            finger_gaps += 1
            cv2.circle(output, far, 8, (0, 0, 255), -1)

    # 손가락 사이 공간 개수 + 1 = 손가락 개수 추정
    fingers = finger_gaps + 1

    # 최대 5개로 제한
    if fingers > 5:
        fingers = 5

    # 결과 표시
    cv2.putText(
        output,
        f"Fingers: {fingers}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3
    )

    return output, mask, fingers, "성공"


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
# -----------------------------
# 한글 경로도 읽을 수 있게 하는 방식
img_array = np.fromfile(file_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    print("이미지를 읽을 수 없습니다.")
    print("선택한 파일 경로:", file_path)
    exit()

# -----------------------------
# 손가락 개수 세기
# -----------------------------
result_img, mask_img, count, message = count_fingers(img)

print(message)
print("추정 손가락 개수:", count)

# -----------------------------
# 화면 출력
# -----------------------------
cv2.imshow("Original Result", result_img)
cv2.imshow("Skin Mask", mask_img)

print("창에서 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()