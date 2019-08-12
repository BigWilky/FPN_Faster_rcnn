# export PYTHONPATH=$PYTHONPATH:/home/lin/projects/models/research:/home/lin/projects/models/research/slim
import numpy as np
import os
import sys
import tensorflow as tf
import time
import grpc
import cv2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from grpc._cython.cygrpc import CompressionAlgorithm
from grpc._cython.cygrpc import CompressionLevel
sys.path.append("")
from libs.box_utils import draw_box_in_img
# import modbus_tk.modbus_tcp as mt #plc 通信
# import modbus_tk.defines as md
import glob as gb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from utils import label_map_util
# from utils import visualization_utils as vis_util
# from PIL import Image, ImageDraw, ImageFont
# PATH_TO_LABELS= "training/labelmap.pbtxt"
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True) 可视化

# import configparser
# cf = configparser.ConfigParser()
# cf.read("/home/wingo/.Wingo/config.yml") 
# secs = cf.sections()

# grpc_host=cf.get("Client", "grpc_host") # ‘127.0.0.1:8500’
# font_dir=cf.get("Client", "FONT")
# plc_host=cf.get("Client","plc_host")
# plc_port=cf.get("Client","plc_port")
grpc_host='127.0.0.1:8500'
tf.app.flags.DEFINE_string('server', grpc_host,
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS
# master = mt.TcpMaster(plc_host, int(plc_port))
# master.set_timeout(5.0) plc 通信



def main():
  video = cv2.VideoCapture(0) # 读摄像头
  options = [('grpc.max_receive_message_length', -1),
                ('grpc.default_compression_algorithm', CompressionAlgorithm.gzip),
                ('grpc.grpc.default_compression_level', CompressionLevel.high)] # 是否启用压缩

  channel = grpc.insecure_channel(FLAGS.server,options)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  # 定义管道，不用变
  request.model_spec.name = 'my_model'
  while(True):
    st = time.time()
    success, frame = video.read()
    
    # Hold_value = master.execute(slave=1, function_code=md.READ_HOLDING_REGISTERS, starting_address=1, quantity_of_x=2, output_value=5)  plc  
    if(success):
      image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 色彩空间变换

      

      retval, buffer = cv2.imencode('.jpg', image_np,[cv2.IMWRITE_JPEG_QUALITY, 95])
      jpg_as_text=bytes(buffer)
      request.inputs['in'].CopyFrom(
          tf.make_tensor_proto(jpg_as_text))#变成tensor
      result = stub.Predict(request, 10.0) #请求
      des=np.array(result.outputs['out'].float_val).reshape(-1,6)
      show_indices = des[:, 1] >= 0.6
      dets_val = des[show_indices]
      show_indices =dets_val[:,0]<=2
      dets_val = dets_val[show_indices]
      final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(frame,
                                                                                boxes=dets_val[:, 2:],
                                                                                labels=dets_val[:, 0],
                                                                                scores=dets_val[:, 1])

      cv2.imshow('test', final_detections)
      print(time.time()-st)
      cv2.waitKey(1)
      
      show_img = cv2.cvtColor(np.array(image_np), cv2.COLOR_RGB2BGR)
      cv2.imshow('test', show_img)
      cv2.waitKey(1)
    else:
      print("camera is not open!!!")
main()






