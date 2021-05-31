import cv2
import mediapipe as mp
import time



class FaceMeshDetector():
    def __init__(self,staticMode = False,maxFaces = 2,minDetection = 0.5,minTrackCon = 0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetection=minDetection
        self.minTrackCon=minTrackCon



        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.minDetection,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=2)

    def findFaceMesh(self,img,draw = True):
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        face = []
        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLMS,self.mpFaceMesh.FACE_CONNECTIONS,self.drawSpec,self.drawSpec)

                for id,lm in enumerate(faceLMS.landmark):
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    #print(id,x,y)
                    face.append([x,y])
        return img,face


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector= FaceMeshDetector()
    while True:
        success, img = cap.read()
        img,face = detector.findFaceMesh(img,False)
        if len(face) != 0:
            print(len(face))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()








































