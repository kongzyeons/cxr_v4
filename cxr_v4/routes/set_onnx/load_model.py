import onnxruntime 
import numpy as np
import torch
from functools import partial
from routes.set_onnx.preprocess import preprocess_image_cls,preprocess_image,draw_image
from routes.set_onnx.detector_utils import non_max_suppression
from routes.set_onnx.preprocess_pnumonia import tranfrom_image




class Model():
    def __init__(self,weights,imgz):
        self.model =  onnxruntime.InferenceSession(weights)
        self.imgz = imgz

        self.model_batch_size = self.model.get_inputs()[0].shape[0]
        
        model_h = self.model.get_inputs()[0].shape[2]
        model_w = self.model.get_inputs()[0].shape[3]
        in_w = self.imgz if (model_w is None or isinstance(model_w, str)) else model_w
        in_h = self.imgz if (model_h is None or isinstance(model_h, str)) else model_h
        print("Input Layer: ", self.model.get_inputs()[0].name)
        print("Output Layer: ", self.model.get_outputs()[0].name)
        print("Model Input Shape: ", self.model.get_inputs()[0].shape)
        print("Model Output Shape: ", self.model.get_outputs()[0].shape)
        self.preprocess_func =  partial(preprocess_image, in_size=(in_h, in_h))

    
    def predict_cls_pneu(self,image):
        transform = tranfrom_image(self.imgz,self.imgz)
        img_tensor = transform(image)
        proc_img = img_tensor.unsqueeze(0).numpy()
        model_input =  proc_img
        batch_size = model_input.shape[0] if isinstance(self.model_batch_size, str) else self.model_batch_size
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: model_input})
        pred = np.exp(outputs)
        return pred

    def predict_obj(self,image,confident=0.5,iou_thres=0.5):
        open_cv_image = np.array(image)
        orig_img = open_cv_image[:, :, ::-1]
        proc_img = self.preprocess_func(orig_img)
        proc_img = np.expand_dims(proc_img, axis=0)
        orig_input, model_input = np.array(orig_img), proc_img
        batch_size = model_input.shape[0] if isinstance(self.model_batch_size, str) else self.model_batch_size
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: model_input})

        # print(len(outputs))
        batch_detections = []
        official=True
        if official and len(outputs) >= 1:  
            batch_detections = torch.from_numpy(np.array(outputs[0]))
            batch_detections = non_max_suppression(
                batch_detections, conf_thres=confident, iou_thres=iou_thres, agnostic=False)
        # print(batch_detections)
        labels = batch_detections[0][..., -1].numpy()
        confs = batch_detections[0][..., 4].numpy()
        boxs = batch_detections[0][..., :4].numpy()

        return proc_img,batch_detections,labels,confs,boxs



