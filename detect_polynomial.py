import cv2
import numpy as np
from tensorflow.keras.models import load_model


def preprocess(img):
    """Convert to grayscale, denoise, threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # adaptive threshold handles variable lighting
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )
    return th, gray

def find_boxes(binary_img, img_shape):
    """Find contours that look like worksheet boxes using adaptive thresholds."""
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w
    
    # Use CCOMP to get hierarchy
    cnts, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # First pass: collect all potential boxes with their properties
    candidates = []
    for i, c in enumerate(cnts):
        # Only take top-level contours (no parent)
        if hierarchy[0][i][3] != -1:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # Calculate relative metrics
        area_ratio = area / img_area
        aspect = w / float(h)
        
        # Store candidate with metrics
        candidates.append({
            'contour': c,
            'bbox': (x, y, w, h),
            'area': area,
            'area_ratio': area_ratio,
            'aspect': aspect,
            'perimeter': cv2.arcLength(c, True)
        })
    
    if not candidates:
        return []
    
    # Sort by area to find box-sized contours
    candidates.sort(key=lambda x: x['area'], reverse=True)
    
    # Calculate statistics on the larger contours (likely boxes)
    large_candidates = candidates[:min(10, len(candidates))]
    areas = [c['area'] for c in large_candidates]
    median_area = np.median(areas)
    
    # Filter boxes using adaptive thresholds
    boxes = []
    for cand in candidates:
        area = cand['area']
        area_ratio = cand['area_ratio']
        aspect = cand['aspect']
        x, y, w, h = cand['bbox']
        
        # Area constraints (relative to image)
        # Boxes should be 1-15% of image area
        if area_ratio < 0.01 or area_ratio > 0.15:
            continue
        
        # Should be at least 20% of median large contour
        if area < median_area * 0.2:
            continue
        
        # Aspect ratio - boxes are roughly square
        if aspect < 0.7 or aspect > 1.4:
            continue
        
        # Check for roughly rectangular shape
        epsilon = 0.02 * cand['perimeter']
        approx = cv2.approxPolyDP(cand['contour'], epsilon, True)
        if len(approx) < 4 or len(approx) > 8:
            continue
        
        # Solidity check - boxes should be fairly solid (not too hollow)
        hull = cv2.convexHull(cand['contour'])
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.7:  # boxes should be at least 70% solid
                continue
        
        # Stroke width check - boxes have thicker borders than symbols
        # Check the border thickness by looking at the contour
        mask = np.zeros(binary_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cand['contour']], -1, 255, -1)
        
        # Erode to estimate border thickness
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=2)
        border_pixels = cv2.countNonZero(mask) - cv2.countNonZero(eroded)
        border_ratio = border_pixels / area if area > 0 else 0
        
        # Boxes have relatively thin borders compared to their area
        if border_ratio > 0.5:  # if more than 50% is border, it's likely a symbol
            continue
        
        boxes.append((x, y, w, h))
    
    return boxes

def sort_boxes(boxes):
    """Sort boxes strictly left-to-right."""
    return sorted(boxes, key=lambda b: b[0])


def extract_rois(boxes, gray_img):
    """Extract regions of interest with border cropping (15%) and preprocessing."""
    rois = []

    for (x, y, w, h) in boxes:
        # Original bounds (clipped to image)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gray_img.shape[1], x + w)
        y2 = min(gray_img.shape[0], y + h)

        # Compute 15% margins
        dx = int(0.15 * (x2 - x1))
        dy = int(0.15 * (y2 - y1))

        # Shrink box on all sides
        cx1 = min(x2, x1 + dx)
        cy1 = min(y2, y1 + dy)
        cx2 = max(cx1 + 1, x2 - dx)
        cy2 = max(cy1 + 1, y2 - dy)

        roi = gray_img[cy1:cy2, cx1:cx2]

        # Safety check (skip empty crops)
        if roi.size == 0:
            continue

        # Clean ROI for OCR
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        rois.append(roi)

    return rois


def draw_detected_boxes(img, boxes):
    """Debug visualization with labels."""
    out = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 3)
        cv2.putText(out, str(i), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return out


# ========================
# MAIN PIPELINE
# ========================

def detect_worksheet_boxes(path_to_img):
    """Main detection pipeline."""
    img = cv2.imread(path_to_img)
    if img is None:
        raise ValueError("Could not load image.")
    
    binary, gray = preprocess(img)
    
    # Detect contours that look like boxes
    boxes = find_boxes(binary, img.shape)
    
    # Sort them into reading order
    boxes = sort_boxes(boxes)
    
    # Extract each box region for OCR
    rois = extract_rois(boxes, gray)
    
    # Visualize detected boxes
    debug_img = draw_detected_boxes(img, boxes)
    
    return boxes, rois, debug_img # use rois for OCR

def build_polynomial(detections):
    """
    Converts detections into a 2D array:
    [
      [exp1, exp2, exp3, ...],
      [coef1, coef2, coef3, ...]
    ]
    """

    if not detections:
        return []

    exponents = []
    coefficients = []
    for box in detections:
        if box["role"] == "exponent":
            exponents.append(box["digit"] if box["digit"] != "-" else 0)
        elif box["role"] == "coefficient":
            coefficients.append(box["digit"] if box["digit"] != "-" else 0)
    
    return [
        exponents,
        coefficients
    ]

def preprocess_roi_for_mnist(roi):
    # 1. Grayscale
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    roi = cv2.fastNlMeansDenoising(roi, None, h=10)

    # 3. Resize to MNIST size
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    # 4. Threshold
    try:
        _, roi = cv2.threshold(
            roi, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    except cv2.error:
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)

    # 5. Normalize to [0,1]
    roi = roi.astype("float32") / 255.0

    # 6. Add channel + batch dimensions
    roi = roi.reshape(1, 28, 28, 1)

    return roi


MODEL = load_model("mnist_digit_model.h5")

def extract_polynomial_boxes(path_to_img):
    """
    Detect worksheet boxes, classify digits with CNN,
    and build a structured polynomial representation.
    """
    # 1. Detect boxes + ROIs
    boxes, rois, _ = detect_worksheet_boxes(path_to_img)

    if len(rois) == 0:
        return {
            "num_boxes": 0,
            "detections": [],
            "polynomial": []
        }

    # 2. Load CNN model
    model = MODEL

    detections = []

    # 3. Run inference on each ROI
    for i, roi in enumerate(rois):
        x, y, w, h = boxes[i]
        role = "coefficient" if i % 2 == 0 else "exponent"

        if is_box_empty(roi):
            detections.append({
                "index": i,
                "digit": "-",
                "confidence": 0.0,
                "role": role,
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                }
            })
            continue

        roi_input = preprocess_roi_for_mnist(roi)

        probs = model.predict(roi_input, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))

        detections.append({
            "index": i,
            "digit": digit,
            "confidence": confidence,
            "role": role,
            "bbox": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
        })

    # 4. Build polynomial structure (NEW)
    polynomial = build_polynomial(detections)

    # 5. Return JSON-safe output
    return {
        "num_boxes": len(detections),
        "detections": detections,
        "polynomial": polynomial
    }

def is_box_empty(roi):
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = roi.shape
    total_pixels = h * w

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    white_pixels = cv2.countNonZero(clean)

    # Very small stroke area → empty
    return white_pixels < max(20, total_pixels * 0.005)

# ========================
# Example Usage
# ========================

if __name__ == "__main__":
    result = extract_polynomial_boxes("worksheet_photo3.jpg")

    print(f"Detected {result['num_boxes']} boxes")

    # Print raw detections (optional, good for debugging)
    for d in result["detections"]:
        print(
            f"Box {d['index']}: digit={d['digit']}, "
            f"conf={d['confidence']:.2f}, "
            f"role={d.get('role', 'unknown')}"
        )

    # Pretty-print polynomial
    polynomial = result["polynomial"]

    if not polynomial:
        print("No polynomial detected.")
    else:
        exponents, coefficients = polynomial

        terms = []
        for coef, exp in zip(coefficients, exponents):
            if exp == 0:
                terms.append(str(coef))
            elif exp == 1:
                terms.append(f"{coef}x")
            else:
                terms.append(f"{coef}x^{exp}")

        polynomial_str = " + ".join(terms)
        print("\nDetected polynomial:")
        print(polynomial_str)
