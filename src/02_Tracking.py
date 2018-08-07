
# coding: utf-8

# In[1]:


from imutils.video import WebcamVideoStream
from imutils.video import FPS

import cv2
import imutils
import time


# In[2]:


(major, minor) = cv2.__version__.split(".")[:2]


# In[3]:


(major, minor)


# In[4]:


OPENCV_OBJECT_TRACKERS = {
    'crst': cv2.TrackerCSRT_create,
    'kcf': cv2.TrackerKCF_create,
    'boosting': cv2.TrackerBoosting_create,
    'mil': cv2.TrackerMIL_create,
    'tld': cv2.TrackerTLD_create,
    'medianflow': cv2.TrackerMedianFlow_create,
    'mosse': cv2.TrackerMOSSE_create
}


# In[5]:


tracker = OPENCV_OBJECT_TRACKERS['crst']
initBB = None
video_capture = WebcamVideoStream(src=0).start()
fps = None
while True:
    frame = video_capture.read()
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    
    if initBB is not None:
        (success, box) = tracker.update(frame)
        
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
        fps.update()
        fps.stop()
        
        info = [
            ("Tracker", (str(tracker)).split(" ")[2])
            ("Success", "Yes" if success else "NO"),
            ("FPS", "{:2f}".format(fps.fps())),
        ]
        
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        
        print(initBB)
        tracker.init(frame, initBB)
        fps = FPS().start()
    
    elif key == ord("q"):
        break
vs.stop()
cv2.destroyAllWindows()       

