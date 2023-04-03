from cv2 import cv2
import numpy as np

cap = cv2.VideoCapture(r'video_folder/004.avi')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()


def k_means_classif(img, success, n_clasters):
    """ input img and status of frame success,number_clasters
        return frame"""
    frame = img.copy()
    if not success:
        raise Exception('Ошибка')
    frame = frame.reshape((-1, 3))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.float32(frame)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(frame, n_clasters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    frame = res.reshape((img.shape))
    return frame


def hog_person_detection(img):
    """ input img
        return frame and boxes"""
    frame = cv2.resize(img, (640, 480))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(10, 10))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    return frame, boxes


if __name__ == '__main__':
    cloud_point = []
    if cap.isOpened():

        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_number in range(number_of_frames):
            success, img = cap.read()
            if not success:
                raise Exception('Ошибка')

            # frame = k_means_classif(img,success,8)
            _, boxes = hog_person_detection(img)
            for (xA, yA, xB, yB) in boxes:
                cloud_point.append([xA, yA, xB, yB])
            len_cloud = len(cloud_point)
            for ind,points in enumerate(cloud_point):
                if len_cloud > 8:
                    points_before = points
                    """circle"""
                    try:
                        if points[0] < points_before[0]+5 and points[0] > points_before[0] -5\
                                and points[1] < points_before[1]+5 and points[1] > points_before[1] -5\
                                and points[2] < points_before[2]+5 and points[2] > points_before[2] -20\
                                and points[3] < points_before[3]+5 and points[3] > points_before[3] -5:
                            cv2.circle(img, (points[0] + (points[2] - points[0]) // 2, points[3]), radius=3, color=(0, 255, 0),
                               thickness=2)
                    except: points_before = points

                    """Line """
                    # x1 = points_before[0] + (points_before[2] - points_before[0]) // 2
                    # y1 = points_before[3]
                    # x2 = points[0] + (points_before[2] - points_before[0]) // 2
                    # y2 = points[3]
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                    points_before = points
                else: pass

            cv2.imshow('frame', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()