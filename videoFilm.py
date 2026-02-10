import cv2
import recordData
from consts import cap

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False


while True:

    print(recordData.getTransmitStart())
    print(recordData.getTransmitRestart())
    print(recordData.getTransmitStop())

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera View', frame)

    if recording:
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        recording = not recording
        if recording:
            h, w = frame.shape[:2]
            shooter_speed = recordData.get_shooter_speed()
            angle = 17.75

            filename = f"film/{shooter_speed}_{angle}.mp4"
            out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        else:
            out.release()

    elif recordData.getTransmitStop() or key == ord('q'):
        break

cap.release()
if out: out.release()
cv2.destroyAllWindows()
# 17 163 146 25 255 255