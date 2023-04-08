import numpy
from cv2 import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def load_model():
    """ load model deeplabv3_resnet50"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

def delete_background(pic, mask):
    # split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    # add an alpha channel with and fill all with transparent pixels (max 255)
    a = np.ones(mask.shape, dtype='uint8') * 255
    # merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)
    # create a transparent background
    bg = np.zeros(alpha_im.shape)
    # setup the new mask
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    # copy only the foreground color pixels from the original image where mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground

def hog_person_detection(img):
    """ input img
        return frame and boxes"""
    frame = cv2.resize(img, (320, 240))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(10, 10))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    return frame, boxes

def remove_background(model, input_file):
    input_image = Image.fromarray(input_file)
    try:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        # create batch
        input_batch = input_tensor.unsqueeze(0)

        # move to gpu
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        bin_mask = np.where(mask, 255, background).astype(np.uint8)

        foreground = delete_background(input_image ,bin_mask)
    except: return None,None

    return foreground, bin_mask

if __name__ == "__main__":


    #Video
    cap = cv2.VideoCapture(r'video_folder/003.avi')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()
    deeplab_model = load_model()

    if cap.isOpened():

        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in range(number_of_frames):
            success, img = cap.read()
            img = cv2.resize(img, (320, 240))
            if not success:
                raise Exception('Ошибка')

            foreground, _ = remove_background(deeplab_model, img)
            if foreground is not None:
                _, boxes = hog_person_detection(foreground)
                #not empty
                if boxes !=[[]]:
                    for (xA, yA, xB, yB) in boxes:
                        cv2.circle(img, (xA + (xB - xA) // 2, yB), radius=3, color=(0, 255, 0),
                                   thickness=2)
                        cv2.imshow('frame1', img)
                        cv2.imshow('frame2', foreground)
                else: cv2.imshow('frame1', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()



