import cv2

img = cv2.imread("../images/people_on_street.jpeg")

## Transformar imagem em gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Carregando o modelo
person_classifier_file = '../models/full_body.xml'
person_cascade = cv2.CascadeClassifier(person_classifier_file)

## Gerando previsão
persons = person_cascade.detectMultiScale(gray_img)

## Inserir retângulos e texto nas detecções da imagem
for (x,y,w,h) in persons:
    gray = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),16)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'person',(x, y-10), font, 2.0, (11,255,255), 6)

## Salvar Imagem com detecção
# cv2.imwrite('../results/person_detection.jpeg', img) 

## Mostrar imagem
cv2.imshow('person_detection',img)

## Aguardar até que qualquer tecla seja pressionada
cv2.waitKey(0)