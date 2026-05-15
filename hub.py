import cv2
import tkinter as tk
from tkinter import filedialog
import math
import numpy as np
import sys

# -----------------------------
# 손가락 개수 추정 함수
# -----------------------------
def count_fingers(img):
    # 이미지 크기가 너무 크면 처리가 느리고 인식이 안 될 수 있으므로 리사이징
    height, width = img.shape[:2]
    scaling_factor = 640 / width
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    output = img.copy()

    # 1. BGR → HSV 변환 (조명 변화에 강함)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 피부색 범위 설정 (한국인 피부색에 좀 더 보편적인 범위로 미세 조정)
    lower_skin = np.array([0, 20, 70], dtype="uint8")
    upper_skin = np.array([20, 255, 255], dtype="uint8")

    # 3. 피부색 마스크 생성 및 노이즈 제거
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 4. 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return output, mask, 0, "손을 찾을 수 없습니다."

    # 가장 큰 윤곽선을 손이라고 가정
    hand_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(hand_contour)

    # 손 영역이 너무 작으면 무시 (노이즈 방지)
    if area < 5000:
        return output, mask, 0, "손 영역이 너무 작거나 멀리 있습니다."

    # 5. 볼록 껍질(Convex Hull) 및 볼록 결함(Convexity Defects) 계산
    # 근사화 작업을 통해 노이즈를 줄임
    epsilon = 0.001 * cv2.arcLength(hand_contour, True)
    approx_contour = cv2.approxPolyDP(hand_contour, epsilon, True)
    
    hull = cv2.convexHull(approx_contour, returnPoints=False)
    
    if hull is None or len(hull) < 3:
        return output, mask, 0, "손 모양 분석 실패"

    defects = cv2.convexityDefects(approx_contour, hull)

    finger_gaps = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(approx_contour[s][0])
            end = tuple(approx_contour[e][0])
            far = tuple(approx_contour[f][0])

            # 세 점 사이 거리 계산
            a = math.dist(start, end)
            b = math.dist(start, far)
            c = math.dist(end, far)

            # 코사인 법칙으로 각도 계산 (손가락 사이의 각도는 보통 90도 미만)
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c + 1e-6)) * 180 / math.pi

            # d(깊이) 값이 일정 수준 이상이어야 실제 손가락 사이 골짜기로 인정
            if angle <= 90 and d > 12000:
                finger_gaps += 1
                cv2.circle(output, far, 8, [0, 0, 255], -1) # 골짜기 표시
                cv2.line(output, start, end, [0, 255, 0], 2) # 손가락 연결선

    # 손가락 개수 = 골짜기 개수 + 1 (단, 주먹 쥐었을 때 등 예외 처리)
    fingers = finger_gaps + 1
    if fingers > 5: fingers = 5
    if finger_gaps == 0 and area > 10000: fingers = 1 # 골짜기가 없는데 영역이 크면 보통 엄지만 핀 상태 등

    # 결과 텍스트 삽입
    cv2.putText(output, f"Fingers: {fingers}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    return output, mask, fingers, "분석 성공"

# -----------------------------
# 실행부
# -----------------------------
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="손 사진을 선택하세요",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("사진이 선택되지 않았습니다.")
    sys.exit()

# 한글 경로 포함 이미지 로드
img_array = np.fromfile(file_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    print("이미지를 불러올 수 없습니다.")
    sys.exit()

result_img, mask_img, count, message = count_fingers(img)

print(f"결과 메시지: {message}")
print(f"추정 손가락 개수: {count}")

cv2.imshow("Result (Press any key to exit)", result_img)
cv2.imshow("Skin Detection Mask", mask_img)

cv2.waitKey(0)
cv2.destroyAllWindows()