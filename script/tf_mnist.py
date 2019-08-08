import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import argparse
import logging
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.keras import layers
import tensorflow.keras as K
import numpy as np
import pandas as pd

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras import backend as bk


class BasicModel:
    def __init__(self):
        self.init_model()
    
    # 创建模型
    def init_model(self):
        # Sequential 与 Functional Model均可，这里以Functional Model为例构建
        inputs = K.Input(shape=(28,28), name='input')
        flatten = layers.Flatten()(inputs)
        dense1 = layers.Dense(128, activation='relu')(flatten)
        dense2 = layers.Dense(10, activation='softmax', name='output')(dense1)
        self.model = K.Model(inputs=inputs, outputs=dense2)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return
        
    # 获取模型
    def get_model(self):
        return self.model
    
    # 训练模型
    def train(self, epochs, batch_size, train_x, train_y, model_dir):
        # 为训练添加 checkpoint 存储
        checkpoint_path = model_dir + os.sep + 'model.ckpt'
        save_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=2, callbacks = [save_callback])
        return
    
    # 保存模型
    def save_model(self, model_name):
        path = '/opt/ml/model/{}'.format(model_name) 
        if '14' in tf.__version__:
            # 若tensorflow版本为1.14，使用 export_saved_model 方法导出 Saved Model
            K.experimental.export_saved_model(self.model, path)
        elif '13' in tf.__version__:
            # 若tensorflow版本为1.13，使用 export 方法导出 Saved Model
            K.experimental.export(self.model, path)
        else:
            # 当tensorflow版本低于1.12及以下，Keras还没有相关方法支持，需要自行构建 signature 生成 Saved Model
            bd = builder.SavedModelBuilder(path)
            signature = predict_signature_def(
                inputs={"input": self.model.input}, outputs={"score": self.model.output})
            with bk.get_session() as sess:
                bd.add_meta_graph_and_variables(
                    sess=sess, tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})
                bd.save()
        return
        

        
# 设置打印日志相关信息
def set_up_logger():
    logger = logging.getLogger('runqi_demo')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
    

# Main方法
if __name__ == '__main__':
    
    # 解析脚本传入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_dir', type=str)
    # parse_known_args 当有上面未提到的args传入时，不会报错，而parse_args则会报错
    args,_ = parser.parse_known_args()
    
    # 打印日志
    logger = set_up_logger()
    logger.info('script load successfully!')
    logger.info('tensorflow version is {}'.format(tf.__version__))
    
    # 获取训练数据
    TRAIN_PATH = os.environ.get('SM_CHANNEL_TRAIN')
    train_x = np.load(TRAIN_PATH + os.sep + 'train_x.npy')
    train_y = np.load(TRAIN_PATH + os.sep + 'train_y.npy')
    
    
    # 创建模型
    model = BasicModel()
    
    # 训练模型
    model.train(args.epochs, args.batch_size, train_x, train_y, args.model_dir)
    
    # 保存模型
    # 被保存的模型名称应该仅包含数字，这一点受限于SageMaker的部署只能找到数字格式名称的model
    model.save_model('167213')
    
    # 任务完成
    logger.info('training job finished! ')
    
    