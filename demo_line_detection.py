import cv2
import torch 
import time
import argparse
import numpy as np

from src import util
from net import Net
from src.parameters import Parameters
from src.processing_image import warp_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--option', type=str, default='image', help="demo line detection on 'image' or 'video', default 'image' ", required=True)
    parser.add_argument('-d','--direction', type=str, default="", help='direction of demo video', required=True)
    parser.add_argument('-s','--save_video', type=bool, default=False)
    args = vars(parser.parse_args())
    
    net = Net()
    p = Parameters()
    # load model epoch 34 with total loss is 0.7828
    net.load_model(1,0.8777)

    # read image from folder images test
    if args['option'] == 'image':

        image = cv2.imread(args['direction'])

        image_resized = cv2.resize(image,(512,256))
        cv2.imshow("image",image_resized)

        #x , y are position of points in lines 
        #because previous image is warped -> warp = False
        x , y = net.predict(image_resized, warp = False)
        print(x, y)
        image_points_result = net.get_image_points()
        cv2.imshow("points", image_points_result)
        cv2.imwrite("result.png",image_points_result)
        cv2.waitKey()
    if args['option'] == 'video':
        cap = cv2.VideoCapture(args['direction'])
        if args['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('result-point.avi', fourcc, 30, (512,256))
        while cap.isOpened():
            prevTime = time.time()
            ret, image = cap.read()
            t_image = cv2.resize(image,(512,256))
            x , y = net.predict(t_image)
            # fits = np.array([np.polyfit(_y, _x, 1) for _x, _y in zip(x, y)])
            # fits = util.adjust_fits(fits)
            image_points = net.get_image_points()
            # mask = net.get_mask_lane(fits)
            cur_time = time.time()
            fps = 1/(cur_time - prevTime)
            s = "FPS : "+ str(fps)
            # image_lane = net.get_image_lane()
            cv2.putText(image_points, s, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow("image",image_points)
            out.write(image_points)
            
            key = cv2.waitKey(1)
            if not ret or key == ord('q'):
                break
        
        out.release()
