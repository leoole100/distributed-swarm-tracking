# %%
import cv2
cap = cv2.VideoCapture(0)

# %%
# backSub = cv2.createBackgroundSubtractorKNN()

backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if frame is None: break

    mask = backSub.apply(frame)

    cv2.imshow("frame", frame)
    cv2.imshow("backSub", mask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
# %%
