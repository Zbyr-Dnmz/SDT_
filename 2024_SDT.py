import cv2
import numpy as np

mesafe_sensor = 0.15
gercek__cizgi_boyut = 5

def process_frame(image):
    height, width = image.shape
    olcek_faktoru = width / mesafe_sensor
    sanal_cizgi_boyutu = int(gercek__cizgi_boyut * olcek_faktoru / 100)

    _, thresh_image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
    
    tolerans_ust_min_y = int(height / 4 - max(5, int(min(height, width) * 0.02)))
    tolerans_ust_max_y = int(height / 4 + max(5, int(min(height, width) * 0.2)))
    tolerans_alt_min_y = int(height / 4 * 3 - max(5, int(min(height, width) * 0.02)))
    tolerans_alt_max_y = int(height / 4 * 3 + max(5, int(min(height, width) * 0.2)))
    tolerans_min_x = int((width - sanal_cizgi_boyutu) / 2) - max(20, int(min(height, width) * 0.02))
    tolerans_max_x = int((width + sanal_cizgi_boyutu) / 2) + max(20, int(min(height, width) * 0.02))

    ust_alt_sol_hesap = int(width * 0.2)
    ust_alt_sag_hesap = int(width * 0.8)

    ust_sol = (ust_alt_sol_hesap, 0)
    ust_sag = (ust_alt_sag_hesap, 0)
    
    alt_sol = (ust_alt_sol_hesap, int(height / 3 * 2))
    alt_sag = (ust_alt_sag_hesap, int(height / 3 * 2))

    sorgu = [ust_sol, ust_sag]

    global kontrol

    if kontrol == 1:
        x, y, w, h = tolerans_min_x, tolerans_alt_min_y, tolerans_max_x, tolerans_alt_max_y
    else:
        x, y, w, h = tolerans_min_x, tolerans_ust_min_y, tolerans_max_x, tolerans_ust_max_y

    cropped_image = thresh_image[y:h, x:w]

    for sensor in sorgu:
        sx, sy = sensor
        if thresh_image[sy, sx] == 0:
            kontrol = 1
        if (thresh_image[alt_sol[1], alt_sol[0]] == 0) or (thresh_image[alt_sag[1], alt_sag[0]] == 0):
            kontrol = 0

    return thresh_image, cropped_image

def check_border(image):
    height, width = image.shape
    
    left_border = image[:, 0]
    right_border = image[:, -1]
    top_border = image[0, :]
    bottom_border = image[-1, :]
    
    left_black = np.any(left_border == 0)
    right_black = np.any(right_border == 0)
    top_black = np.any(top_border == 0)
    bottom_black = np.any(bottom_border == 0)

    directions = {
        'left': left_black,
        'right': right_black,
        'top': top_black,
        'bottom': bottom_black
    }
    return directions

def pixel_color_ratio(image, point1, point2):
    height, width = image.shape
    x1, y1 = point1
    x2, y2 = point2

    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    black_pixels = 0

    for x, y in points:
        if y == height:
            y -= 1
        pixel_value = image[y, x]
        if pixel_value == 0:
            black_pixels += 1

    total_pixels = len(points)
    black_ratio = black_pixels / total_pixels

    return black_ratio

def find_slope(image):
    height, width = image.shape
    
    upper_black_indices = np.where(image[1, :] == 0)
    lower_black_indices = np.where(image[height - 1, :] == 0)

    if len(upper_black_indices[0]) == 0 or len(lower_black_indices[0]) == 0:
        return 

    upper_min_x = np.min(upper_black_indices)
    upper_max_x = np.max(upper_black_indices)
    
    lower_min_x = np.min(lower_black_indices)
    lower_max_x = np.max(lower_black_indices)

    min_egim = (height - 0) / (upper_min_x - lower_min_x) if upper_min_x != lower_min_x else float('inf')
    min_angle_rad = np.arctan(min_egim)
    min_angle_deg = int(np.degrees(min_angle_rad))
    
    max_egim = (height - 0) / (upper_max_x - lower_max_x) if upper_max_x != lower_max_x else float('inf')
    max_angle_rad = np.arctan(max_egim)
    max_angle_deg = int(np.degrees(max_angle_rad))
    
    min_slope = pixel_color_ratio(image, (upper_min_x, 0), (lower_min_x, height))
    max_slope = pixel_color_ratio(image, (upper_max_x, 0), (lower_max_x, height))
    
    if max(min_slope, max_slope) == min_slope:
        slope = min_angle_deg
    elif max(min_slope, max_slope) == max_slope:
        slope = max_angle_deg
        
    if slope > 0 and slope != 90:
        slope = 90 - slope 
        return f"{slope}, sol"
    elif slope == 90:
        slope = 0
        return f"{slope}, düz"
    elif slope < 0 and slope != 90:
        slope = -(90 + slope)
        return f"{slope}, sağ"
 
cap = cv2.VideoCapture(0)

kontrol = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_image, cropped_image = process_frame(gray_frame)

    borders = check_border(thresh_image)
    directions = []
    if borders['bottom']:
        if borders['left']:
            directions.append("Sol")
    
        if borders['right']:
            directions.append("Sağ")
    
        if borders['top']:
            directions.append("Yukarı")
    else:
        directions.append('Yol Yok')

    slope_direction = find_slope(cropped_image)
    print(f"Yön: {slope_direction}, {', '.join(directions)}")
    
    cv2.imshow('Original Image', thresh_image)
    cv2.imshow('Cropped Image', cropped_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()