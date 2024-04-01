import cv2

# parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 69, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX

# reading class name from text file
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#  setting up opencv net
yoloNet = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Camera
def ObjectDetector(image):
    classes, scores, boxes = model.detect(
        image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(image, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1]-10), fonts, 0.5, ORANGE, 2)

camera = cv2.VideoCapture(0)
counter = 0
capture = False
number = 0
while True:
    ret, frame = camera.read()

    original = frame.copy()
    ObjectDetector(frame)
    cv2.imshow('original', original)

    print(capture == True and counter < 10)
    if capture == True and counter < 10:
        counter += 1
        cv2.putText(
            frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, ORANGE, 2)
    else:
        counter = 0

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        capture = True
        number += 1
        cv2.imwrite(f'ReferenceImages/image{number}.png', original)
    if key == ord('q'):
        break
cv2.destroyAllWindows()