# Contador-Objetos-IA
Utilizando dos conceitos, conhecimentos e ferramentas da disciplina de Redes Neurais, para desenvolver uma IA capaz de realizar a contagem e rastreamento de objetos, com auxílio de uma câmera.


### Participação no desenvolvimento do sistema de contagem de estudantes:

Contribuir na elaboração do código para análise de imagens e detecção de objetos, utilizando frameworks de IA e visão computacional como TensorFlow, PyTorch e OpenCV.

### Estudo de redes neurais convolucionais e detecção de objetos:
Aprimorar conhecimentos em redes neurais convolucionais (CNNs) e técnicas de detecção de objetos, como YOLO, para aplicar no projeto.

### Implementação de funcionalidades de detecção e contagem de objetos:
Desenvolver e otimizar funcionalidades para identificar, contar e monitorar os objetos nas imagens, garantindo alta precisão e desempenho.

### Pesquisa de técnicas de pré-processamento de imagens: 
Investigar e aplicar métodos de aprimoramento de imagens, como normalização de cores, ajuste de brilho e contraste, para melhorar a qualidade do treinamento do modelo.

### Aplicando YOLOv8n ao código

```
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
model = YOLO("yolov8n.pt") 
```

-> IMPORTANTE: modelo apresenta alguns erros, talvez tentar outra versão do YOLO ou treinar um modelo próprio seria o ideal, utilizando os locais de entrada e saida de alunos.

-> Lembrete: enfrentei alguns erros de dependencias ao instalar o ultralytics, descobri que é necessário ter as build tools do C/C++.

Link para o problema: https://github.com/ultralytics/ultralytics/issues/14948

-> Lembrete 2: Não foi testado em tempo real, apenas utilizando alguns videos, dentre eles o que subido aqui por ter maior taxa de acerto.

## Atualizações futuras: 

Melhorar o modelo e aplicação em tempo real.

## Videos 

-> Subirei em breve videos de como está sendo feito a análise pelo modelo.
