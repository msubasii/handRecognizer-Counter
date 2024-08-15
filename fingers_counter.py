import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate = (4, 2)

while True:
    success, img = cap.read()

    #mirror the image
    img = cv2.flip(img, 1)
    #convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandmarks = results.multi_hand_landmarks

    if multiLandmarks:
        handPoints = []
        for handLms in multiLandmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            for idx, lm in enumerate(handLms.landmark):
                #print(idx, lm)
                h, w, c = img.shape
                cx,cy =int(lm.x*w), int(lm.y*h)
                handPoints.append((cx, cy))

            if len(handPoints) >= 21:
                for point in handPoints:
                    cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)
                
                upCount=0
                for coordinate in fingerCoordinates:
                    if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                        upCount += 1
                if handPoints[thumbCoordinate[0]][0] < handPoints[thumbCoordinate[1]][0]:
                    upCount += 1

                cv2.putText(img, str(upCount), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
        break
cap.release()
cv2.destroyAllWindows()

