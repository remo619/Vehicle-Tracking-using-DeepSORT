import cv2
import numpy as np
from PIL import Image
import time
import tflite_runtime.interpreter as tflite
#import deepsort
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/mobilenet_v2.tflite'
        self.classes_path = 'model_data/classes.txt'
        self.score = 0.5
        self.iou = 0.5
        self.class_names = self._get_class()
        # fixed size or (None, None)
        self.boxes, self.scores, self.classes = self.detect_image()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def detect_image(self, image):
        
        interpreter = tflite.Interpreter(model, num_threads=4)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        
        self.model_image_size = (width, height)
        boxed_image = letterbox_image(image, self.model_image_size)

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        image_data = np.array(image_data, dtype='float32')
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        detected_boxes = interpreter.get_tensor(output_details[0]['index'])
        detected_classes = interpreter.get_tensor(output_details[1]['index'])
        detected_scores = interpreter.get_tensor(output_details[2]['index'])
        num_boxes = interpreter.get_tensor(output_details[3]['index'])

        return_boxes = []
        return_scores = []
        return_class_names = []

        for i, c in reversed(list(enumerate(detected_classes))):
            predicted_class = self.class_names[c]
            box = detected_boxes[i]
            score = detected_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxes.append([x, y, w, h])
            return_scores.append(score)
            return_class_names.append(predicted_class)

        return return_boxes, return_scores, return_class_names


def track():
    #Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    #DeepSORT
    model = "model_data/mars-small128.pb"
    encoder = gdet.create_box_encoder(model, batch_size =1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    cap = cv2.VideoCapture("videos/test.mp4")
    w = int(cap.get(3))
    h = int(cap.get(4))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))
    frame_index = -1
    fps = 0.0

    while True:
        ret, frame = cap.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = YOLO.detect_image(image)

        features = encoder(frame, boxes)
        detections = [Detection(bbox, confidence, clas, feature) for bbox, confidence, clas, feature in zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 1)        
        
        
        cv2.imshow('', frame)
        
        out.write(frame)
        frame_index = frame_index + 1
        fps = (fps + (1./(time.time()-t1))) / 2
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    track()