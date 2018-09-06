import math
import numpy as np
import cv2
from skimage.measure import label, regionprops
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier

def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return (X - x, Y - y)


def unit(v):
    x, y = v
    mag = length(v)
    return (x / mag, y / mag)


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return (x * sc, y * sc)


def add(v, w):
    x, y = v
    X, Y = w
    return (x + X, y + Y)


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)


def pnt2line2(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)

def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def pripremaTrainData(data):
    for i in range(0, len(data)):
        img = cv2.inRange(data[i].reshape(28, 28), 150, 255)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        regions = regionprops(label(closing))

        miny = regions[0].bbox[0]
        minx = regions[0].bbox[1]
        maxy = regions[0].bbox[2]
        maxx = regions[0].bbox[3]

        for region in regions:
            if region.bbox[0] < miny:
                miny = region.bbox[0]
            if region.bbox[1] < minx:
                minx = region.bbox[1]
            if region.bbox[2] > maxy:
                maxy = region.bbox[2]
            if region.bbox[3] > maxx:
                maxx = region.bbox[3]

        height = maxy - miny
        width = maxx - minx

        newImg = np.zeros((28, 28))

        newImg[0:height, 0:width] = newImg[0:height, 0:width] + img[miny:maxy, minx:maxx]

        data[i] = newImg.reshape(1, 784)

def findLineCoords(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, None, 3)
    minLineLength = 50
    maxLineGap = 20
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, minLineLength, maxLineGap)
    distance_max = 0
    x1 = ""
    y1 = ""
    x2 = ""
    y2 = ""
    if lines is not None:
        for i in range(0, len(lines)):
            line = lines[i][0]
            distance = math.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
            if distance > distance_max:
                x1 = line[0]
                y1 = line[1]
                x2 = line[2]
                y2 = line[3]
                distance_max = distance

    return x1, y1, x2, y2

def getImageNmbr(bbox, img):
    min_row = bbox[0]
    height = bbox[2] - min_row
    min_col = bbox[1]
    width = bbox[3] - min_col
    rangeX = range(0, height)
    rangeY = range(0, width)
    img_number = np.zeros((28, 28))
    for x in rangeX:
        for y in rangeY:
            img_number[x, y] = img[min_row + x - 1, min_col + y - 1]
    return img_number


def addnumber(brojevi, broj, frameN):

    for a in brojevi:
        if a[0] == broj and frameN - a[1] < 10:
            brojevi.remove(a)
            brojevi.append((broj, frameN))
            return False
    brojevi.append((broj, frameN))
    return True


if __name__ == '__main__':

    mnist = fetch_mldata('MNIST original')
    train = mnist.data
    pripremaTrainData(train)

    knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(train, mnist.target)
    videoNum = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    kernel = np.ones((1, 1), np.uint8)
    close_kernel = np.ones((4, 4), np.uint8)
    for i in videoNum:
        cap = cv2.VideoCapture("video-" + str(i) + ".avi")
        frameN = 0
        brojac = 0
        brojevi = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frameN % 5 != 0:
                frameN += 1
                continue
            if frameN < 1:
                x1, y1, x2, y2 = findLineCoords(frame)
                #print('Koordinate linije: ')
                #print(x1)
                #print(y1)
                #print(x2)
                #print(y2)
                #print('------------------------------')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            closing = cv2.morphologyEx(cv2.inRange(gray, 163, 255), cv2.MORPH_CLOSE, close_kernel)
            regions = regionprops(label(closing))

            for region in regions:
                xx1 = region.bbox[1]
                xx2 = region.bbox[3]
                yy1 = region.bbox[0]
                yy2 = region.bbox[2]
                width = xx2 - xx1
                height = yy2 - yy1

                #cv2.circle(frame, (xx1, yy1), 3, (0, 0, 255), 3)
                dist1, pnt1, r1 = pnt2line((xx2, yy2), (x1, y1), (x2, y2))

                if dist1 < 4 and height >= 10 and width >= 10:
                    img_number = getImageNmbr(region.bbox, gray)
                    num = int(knn.predict(img_number.reshape(1, 784)))
                    if addnumber(brojevi, num, frameN):
                        brojac += 1
                        continue

            frameN += 1
            #cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            #cv2.imshow('frame', frame)

            if cv2.waitKey(1) and 0xFF == ord('q'):
                break

        suma = 0

        for b in brojevi:
            suma += b[0]
        print('video: ' + str(i))
        print('suma: ' + str(suma))
        print('--------------')
        cap.release()
        cv2.destroyAllWindows()
