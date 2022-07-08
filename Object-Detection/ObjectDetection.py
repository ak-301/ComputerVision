import cv2

img = cv2.imread('onePiece/5.jpg')
cap = cv2.VideoCapture(0)

classNames = []
classFile = 'ODF/coco.names'
with open(classFile, 'rt')as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ODF/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'ODF/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, Confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)
    for classIds, confidence, box in zip(classIds.flatten(), Confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
