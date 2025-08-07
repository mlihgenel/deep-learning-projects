from fer import FER 
import cv2
import pandas as pd

results = []
def get_emotion(capture):

    cap = cv2.VideoCapture(capture)
    detector = FER(mtcnn=True)
    global results

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        time = frame_number / fps
        
        faces = detector.detect_emotions(frame)
        for face in faces:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            best_emotion = max(emotions, key=emotions.get) # değerlere göre max olanı alır
            results.append({
                "time": round(time, 2),
                "emotion": best_emotion,
                "score": round(emotions[best_emotion], 3) # sözlükteki o duygunun değeri 
            })
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{best_emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Video Duygu Tespiti", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    
    return results
    
