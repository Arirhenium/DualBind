import os
import pickle
import torch
import numpy as np
import json
import logging
from datetime import datetime

PT_FEATURE_SIZE = 40

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')#正无穷
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def angle(vector1, vector2):
    cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2)+1e-8)
    angle = np.arccos(cos_angle)
    # angle2=angle*360/2/np.pi
    return angle

def area_triangle(vector1, vector2):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vector1, vector2))
    return trianglearea

def area_triangle_vertex(vertex1, vertex2, vertex3):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vertex2 - vertex1, vertex3 - vertex1))
    return trianglearea

def cal_angle_area(vector1, vector2):
    return angle(vector1, vector2), area_triangle(vector1, vector2)

def cal_dist(vertex1, vertex2, ord=2):
    return np.linalg.norm(vertex1 - vertex2, ord=ord)


class Config(object):
    def __init__(self, config, train=True):
        if train:
            self.mode = 'train'
        else:
            self.mode = 'test'

        config_path = os.path.join('config', f'{config}.json')
        with open(config_path, 'r') as f:
            all_cfg = json.load(f)
        if self.mode == 'train':
            self.train_config = all_cfg['train']
        else:
            self.test_config = all_cfg['test']

    def get_mode(self):
        return self.mode

    def get_config(self):
        if self.mode == 'train':
            return self.train_config
        else:
            return self.test_config

    def show_config(self, train=True):
        print('='*50)
        if self.mode == 'train':
            for key, value in self.train_config.items():
                print(f'{key}: {value}')
        else:
            for key, value in self.test_config.items():
                print(f'{key}: {value}')
        print('='*50)


class TrainLogger:
    def __init__(self, args, cfg_name, create=True):
        time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        root = args.get('log_root', 'runs')
        self.run_dir = os.path.join(root, f'{cfg_name}', f'repeat_{args.get("repeat", 0)}', time_tag)
        self.model_dir = os.path.join(self.run_dir, 'models')

        if create:
            create_dir([self.run_dir, self.model_dir])

        self._logger = logging.getLogger(f'TrainLogger_{time_tag}')
        self._logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)
        try:
            fh = logging.FileHandler(os.path.join(self.run_dir, 'train.log'), encoding='utf-8')
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)
        except Exception:
            pass

    def info(self, msg):
        self._logger.info(msg)

    def get_model_dir(self):
        return self.model_dir