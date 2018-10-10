import os
import sys
import logging
import time
def create_logger(root_output_path, cfg, image_set):
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    
    cfg_name = os.path.basename(cfg).split('.')[0]
    config_output_path = os.path.join(root_output_path,'{}'.format(cfg_name))
    if not os.path.exists(config_output_path):
        os.makedirs(config_output_path)
    
    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path,'{}'.format('_'.join(image_sets)))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        head,
        datefmt='%Y-%m-%d %H:%M:%S')
    # 使用FileHandler输出到文件
    fh = logging.FileHandler(os.path.join(final_output_path, log_file))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger, final_output_path
