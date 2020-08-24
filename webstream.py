from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from webcam_detect import YOLO
import threading
import argparse
import datetime
import imutils
import time
import cv2



outputFrame = None
lock = threading.Lock()

# init flask object
app = Flask(__name__)

# init video stream and setup camera sensor
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    """
    Render home page with video stream
    :return:
    """
    return render_template("index.html")

def object_detect():
    """
    Performs object detection on frames from the video camera
    :param frameCount:
    :return:
    """
    global vs, outputFrame, lock

    total = 0

    yolo = YOLO()

    # loop over frames from video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # detect object on our frame
        r_image, ObjectsList = yolo.detect_img(frame)

        with lock:
            outputFrame = r_image.copy()


def generate():
    """
    Encodes output frame as JPEG data to reduce network load and
    ensure faster transmission of frames.
    :yields: output frame in byte format
    """

    global outputFrame, lock

    while True:
        with lock:
            # skip iteration if output frame is not available
            if outputFrame is None:
                continue

            # encode frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure frame was successfully encoded
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """
    returns response generated along with specific media
    :return:
    """
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# parse command line args and launch Flask app
if __name__ == "__main__":
     ap = argparse.ArgumentParser()
     ap.add_argument('-i', '--ip', type=str, required=True, help='ip address of the device')
     ap.add_argument('-o', '--port', type=int, required=True, help='ephemeral port number of the server (1024 to 65535)')
     args = vars(ap.parse_args())

     # start thread for object detection
     t = threading.Thread(target=object_detect)
     t.daemon = True
     t.start()

     # start flask app
     app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=True)

# release video streamer
vs.stop()
