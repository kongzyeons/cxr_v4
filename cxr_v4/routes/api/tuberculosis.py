import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import io
from io import BytesIO
import base64
import json
import requests

from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter,FastAPI, File, Body, UploadFile, Request,Response,Form
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from fastapi.responses import FileResponse

from routes.set_onnx.load_model import Model
from routes.set_onnx.preprocess import preprocess_image_cls,preprocess_image,draw_image
from routes.set_onnx.detector_utils import non_max_suppression


class DictList(dict):
    def __setitem__(self, key, value):
        try:
            # Assumes there is a list on the key
            self[key].append(value)
        except KeyError: # If it fails, because there is no key
            super(DictList, self).__setitem__(key, value)
        except AttributeError: # If it fails because it is not a list
            super(DictList, self).__setitem__(key, [self[key], value])

dict_class={'Tuberculosis ': 0}
list_class = list(dict_class.keys())
colors = ["green" for i in range(len(list_class))]

print('Start Run Api Tuberculosis ....')

onnx_path ="routes/models/model_obj_tuberculosis /tuberculosis_obj.onnx"
model_tb_obj = Model(onnx_path,640)

router = APIRouter()

@router.post("/predict_tuberculosis_base64", status_code = 201)
async def get_base64(file_base64: str = Form(...),confident: float = Form(...)):
    try:
        # confident = 0.5
        # iou_thres = 0.55
        if 0<=confident<=1:
            print(f'confident : {confident}')
            image = Image.open(BytesIO(base64.b64decode(file_base64))).convert("RGB")
            image_cv = np.array(image)
            proc_img,batch_detections,labels,confs,boxs = model_tb_obj.predict_obj(image,confident=confident)
            if len(boxs) != 0:
                max_pred = max([batch_detections[0][i][4] for i in range(len(batch_detections[0]))])
                img_result, result = draw_image(image, batch_detections,list_class, proc_img, colors)
                ## CV to base64
                img_result = np.array(img_result) 
                img_result = img_result[:, :, ::-1]
                img_base64 = base64.b64encode(cv2.imencode('.jpg', img_result)[1]).decode()
                img_base64 = {'img_base64': img_base64}
                img_base64 = json.dumps(img_base64,ensure_ascii=False, indent=4)
                img_base64 = json.loads(img_base64)

                return {
                    "status" : "SUCCESS",
                    "output_detail" :  {
                        "dectect_count": f'{len(batch_detections[0])}',
                        "Tuberculosis":f"{100*max_pred:.2f}%",
                        "img_base64":img_base64['img_base64']
                    }
                    }

            img_base64 = base64.b64encode(cv2.imencode('.jpg', image_cv)[1]).decode()
            img_base64 = {'img_base64': img_base64}
            img_base64 = json.dumps(img_base64,ensure_ascii=False, indent=4)
            img_base64 = json.loads(img_base64)

            return {
                "status" : "SUCCESS",
                "output_detail" : {
                    "dectect_count": f'{0}',
                    "Tuberculosis":f"{0}",
                    "img_base64":img_base64['img_base64']
                }
                }


        return {"status" : "Error","detail" : "0<=Confident<=1 "} 

    except:
        print("Error: Please read file jpg/png ")
        return {
            "status" : "Error",
            "detail" : "Please read file jpg/png ",
            } 

@router.post("/predict_tuberculosis_image", status_code = 201)
async def get_image(file_image: bytes = File(...),confident: float = Form(...)):
    try:
        # confident = 0.5
        # iou_thres = 0.55
        if 0<=confident<=1:
            print(f'confident : {confident}')
            image = Image.open(io.BytesIO(file_image)).convert("RGB")
            image_cv = np.array(image)
            proc_img,batch_detections,labels,confs,boxs = model_tb_obj.predict_obj(image,confident=confident)
            

            if len(boxs) != 0:
                max_pred = max([batch_detections[0][i][4] for i in range(len(batch_detections[0]))])
                img_result, result = draw_image(image, batch_detections,list_class, proc_img, colors)
                ## CV to base64
                img_result = np.array(img_result) 
                img_result = img_result[:, :, ::-1]
                img_base64 = base64.b64encode(cv2.imencode('.jpg', img_result)[1]).decode()
                img_base64 = {'img_base64': img_base64}
                img_base64 = json.dumps(img_base64,ensure_ascii=False, indent=4)
                img_base64 = json.loads(img_base64)
                
                return {
                    "status" : "SUCCESS",
                    "output_detail" :  {
                        "dectect_count": f'{len(batch_detections[0])}',
                        "Tuberculosis":f"{100*max_pred:.2f}%",
                        "img_base64":img_base64['img_base64']
                    }
                    }

            img_base64 = base64.b64encode(cv2.imencode('.jpg', image_cv)[1]).decode()
            img_base64 = {'img_base64': img_base64}
            img_base64 = json.dumps(img_base64,ensure_ascii=False, indent=4)
            img_base64 = json.loads(img_base64)

            return {
                "status" : "SUCCESS",
                "output_detail" : {
                    "dectect_count": f'{0}',
                    "Tuberculosis":f"{0}",
                    "img_base64":img_base64['img_base64']
                }
                }


        return {"status" : "Error","detail" : "0<=Confident<=1 "} 

    except:
        print("Error: Please read file jpg/png ")
        return {
            "status" : "Error",
            "detail" : "Please read file jpg/png ",
            } 

  