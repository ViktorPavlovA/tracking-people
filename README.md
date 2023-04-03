# Tracking people in real-time



- I used opnecv and numpy with out any CNN architecture.
-  It wasn't a great idea, but i have only five hours to do that:'(
- Сlassifier was `HOGDescriptor`

## **TO DO LIST**

- Need to do filter function
- maybe try to train `yolo CNN` in black and white pictures, also maybe make a shot in way of `k-means` picture,because where are a lot differnet reflections of people.
- by the way, also we can collecte the points of middle boxes and also predict with using `NN with LSTM `. Hello Pytorch.

## how it looks like right now
![alt text](https://i.ibb.co/VgDJ4Cs/photo-2023-04-02-21-58-01.jpg)

Yeap , it not looks sharp and also need apply method: `cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)`
 
