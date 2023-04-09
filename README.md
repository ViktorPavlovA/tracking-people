# Tracking people in real-time



- I used opnecv and numpy with out any CNN architecture.
- Сlassifier was `HOGDescriptor`

## **TO DO LIST**

- Need to do filter function - `done`
- ~~ maybe try to train `yolo CNN` in black and white pictures, also maybe make a shot in way of `k-means` picture,because where are a lot differnet reflections of people. ~~
- ~~by the way, also we can collecte the points of middle boxes and also predict with using `NN with LSTM `. Hello Pytorch.~~
- ~~ also  need try segmentation CNN like Unet, but first need to do is make more than 12 mask. After it should make augmentation. ~~ I used pytorch for segmentation, but `HOGDescriptor` couldn't find man on img then he was far from camera.

## how it looks like right now
![alt text](https://i.ibb.co/VgDJ4Cs/photo-2023-04-02-21-58-01.jpg)

## how it looks like with kmeans
![alt text](https://i.ibb.co/6FtNkKm/iz2-LDu-E-JTs.jpg)

### Final_script.py

![alt text](https://i.ibb.co/6FtNkKm/iz2-LDu-E-JTs.jpg)

It is a final version with fine filter according on data that i grabbed from image

 
