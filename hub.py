import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import math
import numpy as np
import sys

def count_fingers(img):
    try:
        # 1. 이미지 크기 표준화 (너무 크면 연산 오류 발생)
        height, width = img.shape[:2]
        scaling_factor = 640 / width
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        output = img.copy()

        # 2. 피부색 검출 (HSV 색공간 활용)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 3. 노이즈 제거 (모폴로지 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # 4. 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return output, mask, 0, "손을 찾을 수 없습니다."

        hand_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand_contour) < 5000:
            return output, mask, 0, "영역이 너무 작습니다."

        # 5. 볼록 결함(Convexity Defects) 분석
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull)

        finger_gaps = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])

                # 변의 길이 계산
                a = math.dist(start, end)
                b = math.dist(start, far)
                c = math.dist(end, far)

                # 코사인 법칙으로 손가락 사이 각도 계산
                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c + 1e-6)) * 180 / math.pi

                # 각도가 90도 미만이고 깊이가 충분할 때 손가락 사이로 인정
                if angle <= 90 and d > 10000:
                    finger_gaps += 1
                    cv2.circle(output, far, 8, [0, 0, 255], -1)

        fingers = min(finger_gaps + 1, 5)
        cv2.putText(output, f"Fingers: {fingers}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        return output, mask, fingers, "성공"
    except Exception as e:
        return img, None, 0, f"오류 발생: {str(e)}"

# --- 메인 실행부 ---
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True) # 파일 선택창을 맨 앞으로 가져옴

file_path = filedialog.askopenfilename(
    title="손 사진을 선택하세요",
    filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("파일이 선택되지 않았습니다.")
    sys.exit()

try:
    # 한글 경로 지원을 위한 로드 방식
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("이미지 파일을 읽을 수 없습니다.")

    result_img, mask_img, count, message = count_fingers(img)

    print(f"결과: {message} | 추정 개수: {count}")

    cv2.imshow("Result", result_img)
    if mask_img is not None:
        cv2.imshow("Mask", mask_img)

    print("창을 닫으려면 아무 키나 누르세요.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    messagebox.showerror("에러", f"프로그램 실행 중 문제가 발생했습니다:\n{e}")