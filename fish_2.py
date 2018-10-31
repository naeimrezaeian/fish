#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import deque
import numpy as np
import argparse
import imutils
import cv2
from collections import deque
import time
import sys

def check_inside(c, f, t):
    return((c - f > 1) and (c - f < (t - 1)))

def lindist(p1, p2):
    return np.linalg.norm((p1[0] - p2[0], p1[1] - p2[1]))
def angle(p1, p2, p3):
    (u, v, w) = np.array(p1), np.array(p2), np.array(p3)

    u1 = (v - u) / np.linalg.norm(v - u)
    v1 = (w - v) / np.linalg.norm(w - v)
    return np.degrees(np.arccos(np.dot(u1, v1)))

class TimeCount(object):
    def __init__(self, total):
        self._total = total
        self._start = time.time()

    def show(self, count):
        if count == 0:
            count += 1

        deltat = time.time() - self._start
        stept = deltat / float(count)
        finalt = float(self._total - count) * stept

        back = "\b" * 75

        deltats = time.strftime("%H:%M:%S", time.gmtime(deltat))
        finalts = time.strftime("%H:%M:%S", time.gmtime(finalt))

        sys.stderr.write("%sFrame: %5d/%5d (%5.2f%%), Ellap: %s, Expec: %s" % (back, count, self._total, (float(count) / float(self._total)) * 100.0, deltats, finalts))
        sys.stderr.flush()

        
camera = cv2.VideoCapture('f1.mp4')
lumth=200
frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width  = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_count  = camera.get(cv2.CAP_PROP_FRAME_COUNT)

print (frame_width,frame_height)
frame_height=432
frame_width=576
mx=260
my=140
mw=200
mh=175

last_head = None
last_tail = None
f = -1
txt = ""
pts = deque(maxlen=64)
tcount = TimeCount(frame_count)

mask = np.uint8(np.zeros((int(frame_height), int(frame_width))))
mask[my:my + mh, mx:mx + mw] = 255
                
while True:
    f += 1
    # grab the current frame
    ret, frame = camera.read()
    if(not ret):
        break

    if(f % 100 == 0):
        tcount.show(f)
    frame = imutils.resize(frame, width=int(frame_width))    
    #cv2.imwrite("jpg/frame%d.jpg" % f, frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    [hue, sat, lum] = cv2.split(hsv)
    
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    (ret, lum_bin) = cv2.threshold(lum, lumth, 255, cv2.THRESH_BINARY_INV)
    lum_bin = np.bitwise_and(lum_bin, mask)
    (dummy,blobs, dummy) = cv2.findContours(lum_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    blobs = sorted(blobs, key=lambda x: -len(x))
    if (len(blobs) > 0) and (np.size(blobs[0]) > 100):
        blob = blobs[0]

        small_mask = np.uint8(np.ones(np.shape(frame)[:2])) * 0
        cv2.fillConvexPoly(small_mask, blob, 255)

        moments = cv2.moments(small_mask)
        centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        
        dists = map(lambda p: lindist(p[0], centroid), blob)
        tail = tuple(blob[dists.index(max(dists))][0])

        dists = map(lambda p: lindist(p[0], tail), blob)
        head = tuple(blob[dists.index(max(dists))][0])

        # doesn't consider when the fish touches the limits
        if(check_inside(head[0], mx, mw) and check_inside(head[1], my, mh) and check_inside(tail[0], mx, mw) and check_inside(tail[1], my, mh)):
            body_angle = angle(head, centroid, tail)

            # swap the head and the tail when needed
            if (last_head is not None) and (lindist(head, last_head) > lindist(head, last_tail)) and (body_angle > 20):
                (head, tail) = (tail, head)

        txt = "%d\t1\t%d\t%d\t%d\t%d\t%d\t%d\n" % (f, head[0], head[1], centroid[0], centroid[1], tail[0], tail[1])
        
        # store the head and tail for the next frame
        cv2.line(frame, tail,last_tail,  (0, 0, 255), 2)
        #print(last_tail," ",tail)
        last_head = head
        last_tail = tail
        pts.appendleft(tail)
 
    
    cv2.putText(frame, "%d" % f, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)
    cv2.putText(frame, "%dx%d:%dx%d (%d)" % (mx, my, mw, mh, lumth), (mx, my+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    if(txt == ""):
            cv2.line(frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 1)
            cv2.line(frame, (mx, my + mh), (mx + mw, my), (0, 255, 0), 1)
    else:
            cv2.circle(frame, centroid, 2, (0, 255, 0), -1)
            cv2.circle(frame, tail, 2, (255, 0, 0), -1)
            cv2.circle(frame, head, 2, (0, 0, 255), -1)
            # loop over the set of tracked points
        
            
           
    for i in xrange(1, len(pts)):
            
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 1)       
    cv2.imshow('frame',frame)
    cv2.imshow('Tracking Binary',lum_bin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
#cv2.destoryAllWindows()
print(frame_height,frame_height,frame_count)

