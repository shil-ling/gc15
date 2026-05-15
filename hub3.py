import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import math
import numpy as np
import sys


clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"클릭 추가: {(x, y)} / 총 {len(clicked_points)}개")


def load_image_korean_path(file_path):
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def make_user_skin_mask(img, points):
    """
    사용자가 여러 번 클릭한 손 부분 주변 색을 기준으로 마스크 생성
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_img, w_img = img.shape[:2]
    sample_pixels = []

    r = 12

    for x, y in points:
        x1 = max(0, x - r)
        x2 = min(w_img, x + r)
        y1 = max(0, y - r)
        y2 = min(h_img, y + r)

        roi = hsv[y1:y2, x1:x2]

        if roi.size > 0:
            sample_pixels.append(roi.reshape(-1, 3))

    if not sample_pixels:
        return None

    sample_pixels = np.vstack(sample_pixels)

    # 평균보다 중앙값이 튀는 색에 조금 더 강함
    median_hsv = np.median(sample_pixels, axis=0)

    h_mean, s_mean, v_mean = median_hsv

    lower = np.array([
        max(0, h_mean - 14),
        max(20, s_mean - 75),
        max(40, v_mean - 90)
    ], dtype=np.uint8)

    upper = np.array([
        min(179, h_mean + 14),
        min(255, s_mean + 75),
        min(255, v_mean + 90)
    ], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    return mask


def cluster_points_by_x(points, min_distance=35):
    if not points:
        return []

    points = sorted(points, key=lambda p: p[0])
    clusters = [[points[0]]]

    for p in points[1:]:
        if abs(p[0] - clusters[-1][-1][0]) < min_distance:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    result = []

    for cluster in clusters:
        top_point = min(cluster, key=lambda p: p[1])
        result.append(top_point)

    return result


def count_fingers(img, clicked_points):
    try:
        output = img.copy()

        mask = make_user_skin_mask(img, clicked_points)

        if mask is None:
            return output, None, 0, "클릭 지점이 없습니다."

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return output, mask, 0, "손을 찾을 수 없습니다."

        # 클릭한 점들을 많이 포함하는 윤곽선을 손으로 선택
        best_contour = None
        best_score = -1

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 1000:
                continue

            score = 0

            for point in clicked_points:
                inside = cv2.pointPolygonTest(contour, point, False)

                if inside >= 0:
                    score += 1

            # 클릭 포함 수가 같으면 면적 큰 것 우선
            score = score * 100000 + area

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            best_contour = max(contours, key=cv2.contourArea)

        hand_contour = best_contour
        area = cv2.contourArea(hand_contour)

        if area < 3000:
            return output, mask, 0, "손 영역이 너무 작습니다."

        cv2.drawContours(output, [hand_contour], -1, (0, 255, 0), 2)

        M = cv2.moments(hand_contour)

        if M["m00"] == 0:
            return output, mask, 0, "중심점을 계산할 수 없습니다."

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(output, (cx, cy), 8, (255, 0, 255), -1)

        hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
        hull_points = cv2.convexHull(hand_contour, returnPoints=True)

        defects = None

        if hull_indices is not None and len(hull_indices) > 3:
            defects = cv2.convexityDefects(hand_contour, hull_indices)

        finger_gaps = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]

                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])

                a = math.dist(start, end)
                b = math.dist(start, far)
                c = math.dist(end, far)

                if b == 0 or c == 0:
                    continue

                angle = math.degrees(
                    math.acos(
                        max(-1, min(1, (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
                    )
                )

                depth = d / 256

                if angle < 85 and depth > 20 and far[1] < cy + 50:
                    finger_gaps += 1
                    cv2.circle(output, far, 7, (0, 0, 255), -1)

        x, y, bw, bh = cv2.boundingRect(hand_contour)

        min_tip_distance = bh * 0.25
        tip_candidates = []

        for point in hull_points:
            px, py = point[0]
            dist = math.dist((cx, cy), (px, py))

            if py < cy and dist > min_tip_distance:
                if y < py < cy + 10:
                    tip_candidates.append((px, py))

        fingertips = cluster_points_by_x(tip_candidates, min_distance=40)

        for tip in fingertips:
            cv2.circle(output, tip, 10, (255, 0, 0), -1)

        tip_count = len(fingertips)

        if finger_gaps == 0 and tip_count == 0:
            fingers = 0
        elif finger_gaps == 0:
            fingers = tip_count
        else:
            fingers = finger_gaps + 1

        fingers = max(0, min(fingers, 5))

        if area > 0:
            hull_area = cv2.contourArea(hull_points)

            if hull_area > 0:
                solidity = area / hull_area

                if solidity > 0.88 and finger_gaps == 0:
                    fingers = 0

        for point in clicked_points:
            cv2.circle(output, point, 6, (0, 255, 255), -1)

        cv2.putText(
            output,
            f"Fingers: {fingers}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 0, 0),
            3
        )

        cv2.putText(
            output,
            f"Samples: {len(clicked_points)}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        return output, mask, fingers, "성공"

    except Exception as e:
        return img, None, 0, f"오류 발생: {str(e)}"


# -----------------------------
# 메인 실행부
# -----------------------------
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

file_path = filedialog.askopenfilename(
    title="손 사진을 선택하세요",
    filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("파일이 선택되지 않았습니다.")
    sys.exit()

try:
    img = load_image_korean_path(file_path)

    if img is None:
        raise ValueError("이미지 파일을 읽을 수 없습니다.")

    height, width = img.shape[:2]

    if width > 800:
        scale = 800 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    window_name = "Click Hand Areas"

    print("손 부분을 여러 번 클릭하세요.")
    print("Enter 또는 Space: 분석 시작")
    print("R: 클릭 초기화")
    print("ESC: 종료")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        show_img = img.copy()

        cv2.putText(
            show_img,
            "Click hand areas",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        cv2.putText(
            show_img,
            "Enter/Space: Start | R: Reset | ESC: Exit",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

        cv2.putText(
            show_img,
            f"Samples: {len(clicked_points)}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )

        for point in clicked_points:
            cv2.circle(show_img, point, 6, (0, 255, 255), -1)

        cv2.imshow(window_name, show_img)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            print("ESC를 눌러 종료했습니다.")
            cv2.destroyAllWindows()
            sys.exit()

        elif key == ord("r") or key == ord("R"):
            clicked_points.clear()
            print("클릭 지점을 초기화했습니다.")

        elif key == 13 or key == 32:
            if len(clicked_points) == 0:
                print("분석하려면 손 부분을 최소 1번 클릭해야 합니다.")
            else:
                break

    cv2.destroyWindow(window_name)

    result_img, mask_img, count, message = count_fingers(img, clicked_points)

    print(f"결과: {message} | 추정 개수: {count}")

    cv2.imshow("Result", result_img)

    if mask_img is not None:
        cv2.imshow("Mask", mask_img)

    print("창을 닫으려면 아무 키나 누르세요.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    messagebox.showerror("에러", f"프로그램 실행 중 문제가 발생했습니다:\n{e}")