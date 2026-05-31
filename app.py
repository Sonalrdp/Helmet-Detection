from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

project_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(project_dir, 'best.pt')

if os.path.exists(weights_path):
    model_path = weights_path
else:
    model_path = 'yolov8n.pt'
    print(f"WARNING: 'best.pt' not found. Falling back to pretrained {model_path}. "
          "Place 'best.pt' in the project root to use your helmet detector.")

model = YOLO(model_path)


def gen_frames():
    """Generate video frames from the webcam and overlay detection boxes."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open camera.')

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert frame for YOLO processing and run detection
            results = model(frame, imgsz=640, conf=0.35)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = model.names[cls_id] if cls_id in model.names else str(cls_id)

                    # Compliance coloring
                    if label.lower() == 'with helmet':
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # Encode the frame as JPEG to send in the HTTP stream
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.route('/')
def index():
    """Render the dashboard page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Uses multipart streaming to deliver frames."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Receive a browser-captured frame, run YOLO, and return the annotated image."""
    body = request.get_data()
    if not body:
        return Response('No frame data received', status=400)

    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response('Unable to decode image', status=400)

    results = model(frame, imgsz=640, conf=0.35)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id] if cls_id in model.names else str(cls_id)
            color = (0, 255, 0) if label.lower() == 'with helmet' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return Response('Failed to encode annotated frame', status=500)

    return Response(buffer.tobytes(), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
