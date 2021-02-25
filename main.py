import cv2
import pytesseract


# Some transformations
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # noise removal


def remove_noise(image):
    return cv2.medianBlur(image, 3)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # dilation


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


def main():
    # Uncomment this if you don't have tesseract in your PATH
    # pytesseract.pytesseract.tesseract_cmd = "path to tesseract"

    # Loading sample img
    sample = cv2.imread('images/sample.png')

    # Defining a section of the image
    y_top, y_bot = 244, 355
    x_left, x_right = 5, 450
    details = sample[y_top:y_bot, x_left:x_right]
    # OCR
    details_text = pytesseract.image_to_string(details)
    # Showing image section
    cv2.imshow("just the details", details)
    cv2.waitKey(0)
    print(f"details: {details_text}")

    d = pytesseract.image_to_data(sample, output_type=pytesseract.Output.DICT)
    num_boxes = len(d['text'])
    for i in range(num_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            front_img = cv2.rectangle(sample, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"rectangle: ({x},{y}),({x + w},{y + h}) | confidence: {d['conf'][i]} | text: {d['text'][i]}")

    cv2.imshow('sample with rects', sample)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
