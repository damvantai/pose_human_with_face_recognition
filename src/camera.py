import colorsys
import cv2
import dlib
# import face_recognition
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.layers import Input
from keras.models import load_model

import numpy as np
import face_recognition
import os
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from imutils.video import WebcamVideoStream

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        gpu_num = 1

        # model_path = '../logs/human_pose_dataset_1400_416_yolo/trained_weights_final.h5'
        model_path = '../logs/human_pose_dataset_2388_size_416_batch_size_4/human_pose_dataset_2388_size_416_batch_size_4.h5'
        anchors_path = '../model_data/yolo_anchors.txt'
        classes_path = '../model_data/human_pose.txt'
        score = 0.2
        iou = 0.2
        model_image_size = (416, 416)
        self.sess = K.get_session()

        # Get class
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()

        self.class_names = [c.strip() for c in class_names]

        # Anchors
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

        # Load model
        model_path = os.path.expanduser(model_path)
        assert model_path.endswith('.h5'), 'Keras model end with file .h5'

        num_anchors = len(anchors)
        num_classes = len(self.class_names)

        is_tiny_version = num_anchors==6
        try:
            yolo_model = load_model(model_path, compile=False)
        except:
            if is_tiny_version:
                yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes)
            else:
                self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            
            self.yolo_model.load_weights(model_path)
        else:
            self.yolo_model.layers[-1].output_shape[-1] == num_anchors/len(yolo_model.output) * (num_classes + 5), 'Mismatch between model and given anchor and class sizes'
            
        print("{} model, anchors, and classes loaded.".format(model_path))


        face_encodings_in_room = []
        face_names_in_room = []
        known_face_encodings_array = np.load("../data/numpy/known_face_encoding.npy")
        known_face_names = np.load("../data/numpy/known_face_names.npy")

        # Convert nparray -> List to face_encoding
        len_of_array_known_face_names = len(known_face_names)
        known_face_encodings_array = known_face_encodings_array.reshape(len_of_array_known_face_names, 128)
        known_face_encodings = []
        for i in range(len_of_array_known_face_names):
            known_face_encodings.append(known_face_encodings_array[i])

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, anchors, len(self.class_names), self.input_image_shape, score_threshold=score, iou_threshold=iou)
        num_frame = 0
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.video = WebcamVideoStream(src=0).start()

        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        
    def __del__(self):
        self.video.stop()
    
    def get_frame(self):

        model_image_size = (416, 416)
        frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        frame = cv2.flip(frame, 1)
        frame_process = np.copy(frame)

        image = Image.fromarray(frame_process)

        # Process detect hand and recognition furniture
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                     feed_dict={
                                                         self.yolo_model.input: image_data,
                                                         self.input_image_shape: [image.size[1], image.size[0]],
                                                         K.learning_phase(): 0
                                                     })

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            label = '{} {:.2f}'.format(predicted_class, score)
            
            top, left, bottom, right = box  
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
            cv2.putText(frame, label, (left + 6, top + 30), self.font, 1.0, (255, 0, 255), 1)

            
    #         #-------------------------------------------------------#
    #         # Face recognition
    #         crop_img = frame_process[top:bottom, left:right]
    #         # Convert the image from BGR color to RGB to face_recognition use
    #         rgb_frame = crop_img[:, :, ::-1]

    #         # Find all the faces and face encodings in the current frame of video
    #         face_locations = face_recognition.face_locations(rgb_frame)
    #         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
    #         if not face_encodings:
    #             cv2.putText(frame, label, (left + 6, top + 20), font, 1.0, (0, 0, 255), 1)
    #         else:
    #             for (top1, right1, bottom1, left1), face_encoding in zip(face_locations, face_encodings):
    #                 distance = face_recognition.face_distance(known_face_encodings, face_encoding)
    #                 min_distance = np.min(distance)
    #                 index_point_min = np.argmin(distance)
    #                 if min_distance < 0.5:
    #                     name = known_face_names[index_point_min]
    #                     print(name)
    # #                     cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
    #                     label = name + ": " + label
    #                     cv2.putText(frame, label, (left + 6, top + 20), font, 1.0, (0, 0, 255), 1)
    #                 else:
    #                     label = "Unknown" + ": " + label
    #                     cv2.putText(frame, label, (left + 6, top + 20), font, 1.0, (0, 0, 255), 1)
    # #         #-------------------------------------------------------#


        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()