import torch 
import cv2
from net import Net
from src.parameters import Parameters

if __name__ == "__main__":
    net = Net()
    p = Parameters()
    # load model epoch 34 with total loss is 0.7828
    net.load_model(34,0.7828)
    # read image from folder images test
    image = cv2.imread("images_test/2lines-00001086.jpg")

    image_resized = cv2.resize(image,(512,256))
    cv2.imshow("image",image_resized)

    #x , y are position of points in lines 
    #because previous image is warped -> warp = False
    x , y = net.predict(image_resized, warp = False)
    print(x, y)
    image_points_result = net.get_image_points()
    cv2.imshow("points", image_points_result)
    cv2.waitKey()