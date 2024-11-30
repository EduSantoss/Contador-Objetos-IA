import cv2  
import numpy as np
# escadaFacul
video = cv2.VideoCapture('videos_teste/escalator.mp4')
contador = 0
liberado = False

# Ajustar métricas relacionadas a escada

while True:
    ret,img = video.read()
    img = cv2.resize(img,(1100,720),)
    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    x,y,w,h = 490,230,30,150
    imgTh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((8,8), np.uint8)
    imgDil = cv2.dilate(imgTh,kernel,iterations=2)

    recorte = imgDil[y:y+h,x:x+w]
    brancos = cv2.countNonZero(recorte)

    # variavel liberado para evitar nova contagem baseado no fps do video, impedir que a cada frame ele faça uma nova contagem
    if brancos > 4000 and liberado == True:
        contador +=1
    if brancos < 4000:
        liberado = True
    else:
        liberado =False

    if liberado == False:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4) # cor verde com alguem passando
    else:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 255),4) # cor rosa quando nao tem ninguem passando

    cv2.rectangle(imgTh, (x, y), (x + w, y + h), (255, 255, 255), 6) # variavel com threshold para verificar se os pixels brancos estao corretos
    
    # Textos da imagem
    cv2.putText(img,str(brancos),(x-30,y-50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1) # contagem de pontos brancos
    cv2.rectangle(img, (575,155), (575 + 88, 155 + 85), (255, 255, 255), -1) # quadrado branco para desenhar numero de pessoas contadas
    cv2.putText(img, str(contador), (x+100, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5) # numero de pessoas contadas

    if cv2.waitKey(5)&0xFF == ord("q"):
        break
    #print(contador)
    cv2.imshow('video original',img)
    cv2.imshow('video', cv2.resize(imgTh,(600,500)))
    cv2.waitKey(20)