from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt") 
# modelo nao fica tao adequado, talvez o YOLOv8x atue melhor ou o ideal que seria treinar o proprio modelo para a situação de contagem de alunos

# video a ser adicionado, ou utilizar a camera para capturar com o comando 0
cap = cv2.VideoCapture("videos_teste/escalator.mp4")
# cap = cv2.VideoCapture("0")


assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

classes_to_count = [0] # index 0 para pessoas, 2 para carros, ou retirar classe para contar todos no modelo


line_points = [(600,300), (600,600)]

# region_points = [(20, 400), (1080, 400)]  # Para uma linha
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # Para uma regiao retangular
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # Para um poligono

# Para salvar o video que o modelo vai atuar
video_writer = cv2.VideoWriter("videos_teste/escada.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Iniciar contador de objetos
counter = object_counter.ObjectCounter()
counter.set_args(    
     view_img= True,
     reg_pts= line_points, # ou region_points
     classes_names= model.names,
     draw_tracks= True)


# Processamento do video 
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count) 
    # caso quiser contar todas as classes do algoritmo, excluir o parametro classes!!
    
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)
    
    if cv2.waitKey(5)&0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()