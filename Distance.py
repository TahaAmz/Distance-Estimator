import cv2
import torch

# Distance constants
KNOWN_DISTANCE = 40.0  #Cm
MOBILE_WIDTH = 8.0  #Cm

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detection and texts
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
RED = (0, 0, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX

# getting class names from coco.txt
with open("coco.txt", "r") as f:
    class_names = [cName.strip() for cName in f.readlines()]

# connecting to a camera
# url_cap = "http://192.168.62.215:8080"
# frame_cap = cv2.VideoCapture(url_cap + "/video")
frame_cap = cv2.VideoCapture(0)

# setting up opencv net with YOLOv4
yoloNet = cv2.dnn.readNet("yolov4-tiny.cfg", "yolov4-tiny.weights")
yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


# setting up opencv net with YOLOv5


# object detector function
def object_detector(image):
    classes, confidences, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []
    for (classid, confidence, box) in zip(classes, confidences, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid]}"

        # draw rectangle and label on object
        cv2.rectangle(image, box, color, 1)
        cv2.putText(image, label, (box[0], box[1] - 14), FONT, 0.5, color, 2)

        # adding class id
        # 1: class name  2: object width in pixels  3: position where to draw distance
        if classid == 67:  # phone class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])

    return data_list


# focal length finder function
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


# reading the reference image and finding focal length
mobile_ref = cv2.imread("RefImages\image3.png")
mobile_data = object_detector(mobile_ref)
mobile_width_in_pxl = mobile_data[0][1]
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_pxl)

while True:
    ret, frame = frame_cap.read()
    data = object_detector(frame)
    for d in data:
        if d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        distance_txt = round(distance, 2)
        cv2.rectangle(frame, (x, y), (x + 100, y + 20), BLACK, -1)
        cv2.putText(frame, f"Dis: {distance_txt} Cm", (x + 5, y + 13), FONT, 0.35, RED, 1)

    cv2.imshow("frame", frame)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
frame_cap.release()
