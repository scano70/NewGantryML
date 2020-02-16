import pandas as pd
import math
import numpy as np
import time
import os, cv2

imgPATH = '/home/samcano/pyfiles/250x250/line_imgs/'
csvPATH = '/home/samcano/pyfiles/250x250/250x250_labels.csv'
data = pd.read_csv(csvPATH, skipinitialspace=True)
points = [0] * 508
polar = [0] * 508


def makeDF():

    listing = os.listdir(imgPATH)
    imlist = [0] * 508
    imarray = 0
    for i in range(508):
        b = data['External ID'][i] + ".png"
        img_location = imgPATH + b
        for img in listing:
            if(img == b):
                imarray = cv2.imread(img_location)
                break
        imlist[i] = imarray
        data['External ID'][i] = b

    data['Image Array'] = imlist
    return data

def handleCSV():
    #parse csv for x and y values
    for i in range(508):
        data2 = data['Label'][i].split(':')
        points[i] = [0.0 ,0.0 ,0.0 ,0.0]
        x1 = data2[3].split(",")
        y1 = data2[4].split("}")
        x2 = data2[7].split(",")
        y2 = data2[8].split("}")

        #orient labels so origin is bottom left corner
        x1= float(x1[0])
        y1= -1 * (float(y1[0]) - 250.0)
        x2= float(x2[0])
        y2= -1 * (float(y2[0]) - 250.0)

        #load and normalize
        points[i] = [x1/250.0, y1/250.0, x2/250.0, y2/250.0]
        #points[i] = [x1, y1, x2, y2]
        #print points[i]
    data["Label"] = points
    print data.head()

    return points

def handlePolar(choice):
    #convert rectangular points to polar
    if choice == 1:
        for i in range(508):
            polar[i] = [0.0, 0.0]
            slope = (points[i][3] - points[i][1]) / (points[i][2] - points[i][0])
            theta = math.degrees(math.atan2(points[i][3] - points[i][1],points[i][2] - points[i][0]))
            c = points[i][1] - slope*points[i][0]
            r = abs(c) / math.sqrt(slope**2 +1.0)
            polar[i] = [theta/90, r]
            print polar[i]
 
    #convert polar results back to rectangular
    #else:
    return polar




if __name__ == '__main__':    
    makeDF()
    handleCSV()
    #handlePolar(1)
