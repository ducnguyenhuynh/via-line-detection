import cv2
import json 
import torch
import agent
import util
import csaps
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from data_loader import Generator
from parameters import Parameters


p = Parameters()
loader = Generator()
lane_agent = agent.Agent()
lane_agent.load_weights(3, "tensor(0.6211)")
lane_agent.cuda()
lane_agent.evaluate_mode()


def evaluation(loader, lane_agent, index= -1, thresh = p.threshold_point, name = None):
    result_data = deepcopy(loader.test_data)
    progressbar = tqdm(range(loader.size_test//16))
    for test_image, target_h, ratio_w, ratio_h, testset_index, gt in loader.Generate_Test():
        x, y, _ = test(lane_agent, test_image, thresh, index)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        #x_, y_ = find_target(x_, y_, target_h, ratio_w, ratio_h)
        x_, y_ = fitting(x_, y_, target_h, ratio_w, ratio_h)
        result_data = write_result_json(result_data, x_, y_, testset_index)
        # util.visualize_points_origin_size(x_[0], y_[0], test_image[0]*255, ratio_w, ratio_h)
        # print(gt.shape)
        # util.visualize_points_origin_size(gt[0], y_[0], test_image[0]*255, ratio_w, ratio_h)
        progressbar.update(1)
    progressbar.close()
    if name == None:
        save_result(result_data, "test_result.json")
    else:
        save_result(result_data, name)


def test(lane_agent, test_images, thresh = p.threshold_point, index= -1):

    result = lane_agent.predict_lanes_test(test_images)
    torch.cuda.synchronize()
    confidences, offsets, instances = result[index]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)
        
    return out_x, out_y,  out_images

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
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
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
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y

def fitting(x, y, target_h, ratio_w, ratio_h):
    out_x = []
    out_y = []
    count = 0
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h

    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre+0.00001)
                    pre = pre+0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for h in target_h[count]:
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
            if len(temp_x) != len(temp_y):
                print( len(temp_x) , len(temp_y))
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch) 
        count += 1

    return out_x, out_y

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y  

def write_result_json(result_data, x, y, testset_index):
    for index, batch_idx in enumerate(testset_index):
        for i in x[index]:
            result_data[batch_idx]['lanes'].append(i)
            result_data[batch_idx]['run_time'] = 1
    return result_data

def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

if __name__ == "__main__":
    evaluation(loader, lane_agent)