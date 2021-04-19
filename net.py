import cv2
import torch
import numpy as np
from src import util
from src.processing_image import warp_image
from src.hourglass_network import lane_detection_network
from src.parameters import Parameters
from copy import deepcopy
from torch.autograd import Variable



p = Parameters()

class Net(object):
    def __init__(self):
        self.colours = np.array([[78, 142, 255], [204, 237, 221], [92, 252, 255], [92, 255, 195], [159, 150, 255]])
        self.model = lane_detection_network()
        if torch.cuda.is_available():
            self.model.cuda()

    def load_model(self, epoch, loss_value):
        self.model.load_state_dict(torch.load("src/"+p.model_path+str(epoch)+'_tensor('+str(loss_value)+')_'+'lane_detection_network.pkl', map_location='cuda:0'),False)
    
    def predict(self, image, warp = True):
        self.image = image
        if warp:
            self.warped = warp_image(image)
        else:
            self.warped = image
        # self.original = cv2.resize(warped, (p.x_size, p.y_size))
        
        image = np.rollaxis(self.warped, axis=2, start=0)/255.0
        image = np.array([image])

        inputs = torch.from_numpy(image).float() 
        inputs = Variable(inputs).cuda()

        outputs, features = self.model(inputs)

        confidences, offsets, instances = outputs[0]
        features =  features[0]

        # image = deepcopy(image[0])
        # image =  np.rollaxis(image, axis=2, start=0)
        # image =  np.rollaxis(image, axis=2, start=0)*255.0
        # image = image.astype(np.uint8).copy()

        confidence = confidences[0].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[0].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)

        instance = instances[0].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        x, y = generate_result(confidence, offset, instance, p.threshold_point)
        # print("-----------------------------------------")
        self.x, self.y = eliminate_fewer_points(x, y)
        
        return self.x, self.y

    def get_image_points(self):
        result_image = util.draw_points(self.x, self.y, deepcopy(self.warped))
        return result_image

    def get_image_lane(self):
        
        result = cv2.addWeighted(self.image, 1, self.mask, 0.7, 0.3)

        return result

    def get_mask_lane(self, fits):
        warp = np.zeros_like(self.image)
        y = np.linspace(20, 256, 4)

        for i, fit in enumerate(fits[:-1]):
            x_0 = np.array([np.poly1d(fit)(_y) for _y in y ])
            x_1 = np.array([np.poly1d(fits[i+1])(_y) for _y in y])

            pts_left = np.array([np.transpose(np.vstack([x_0, y]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([x_1, y])))])

            color = self.colours[i]

            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(warp, np.int_([pts]), (int(color[0]),int(color[1]),int(color[2])))
        
        self.mask = cv2.warpPerspective(warp, p.inverse_perspective_transform, (warp.shape[1], warp.shape[0]))
        return self.mask

def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0 : 
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                # flag = 0
                # index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 20:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
    
    return x, y


def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i) > 10:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y  

if __name__ == "__main__":
    net = Net()
    net.load_model(34,0.7828)
    image = cv2.imread("images_test/2lines-00000522.jpg")
    image = cv2.resize(image,(512,256))
    x, y = net.predict(image, warp=False)
    print(x, y)