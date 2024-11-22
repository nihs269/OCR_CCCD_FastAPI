import os
import cv2
import sys
import time
import uvicorn
import shutil
import urllib.request

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from get_info import ReadInfo
from DetecInfoBoxes.GetBoxes import GetDictionary
from NamOcr.tool.predictor import Predictor
from NamOcr.tool.config import Cfg as Cfg_vietocr
from config import opt
from urllib.parse import urlparse
from pydantic import HttpUrl


app = FastAPI()
UPLOAD_DIRECTORY = "Img/OriginalImage/"

get_dictionary = GetDictionary(opt)
sys.path.insert(0, 'DetecInfoBoxes')

# Load Ocr model
config_vietocr = Cfg_vietocr.load_config_from_file('NamOcr/config/vgg-seq2seq.yml')
config_vietocr['device'] = 'cpu'

config_vietocr['weights'] = 'Models/OCR_Vehicle_Registration_0.95.pt'
ocr_predictor = Predictor(config_vietocr)

config_vietocr['weights'] = 'Models/seq2seqocr.pth'
default_ocr_predictor = Predictor(config_vietocr)

# Load Yolo model
id_card_yolo_weight = 'Models/CccdYoloV7.pt'
imgsz, stride, device, half, model, names = get_dictionary.load_model(id_card_yolo_weight)
id_card_reader = ReadInfo(imgsz, stride, device, half, model, names, default_ocr_predictor)

vehicle_registration_yolo_weight = 'Models/DangKyXe.pt'
imgsz, stride, device, half, model, names = get_dictionary.load_model(vehicle_registration_yolo_weight)
vehicle_registration_reader = ReadInfo(imgsz, stride, device, half, model, names, ocr_predictor)


@app.post("/cccd")
async def cccd(file: UploadFile = File(...)):
    try:
        st = time.time()
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return_value = id_card_reader.get_id_card_info(file_location)
        os.remove(file_location)
        print("Time: ", time.time() - st)
        return JSONResponse(status_code=201, content={"message": return_value})
    except:
        return JSONResponse(status_code=500, content={"message": "An error occurred"})


@app.get("/cccd_online")
async def cccd_online(path: HttpUrl = Query(...)):
    try:
        st = time.time()
        file_location = os.path.join(UPLOAD_DIRECTORY, str(time.time()) + '.jpg')
        urllib.request.urlretrieve(str(path), file_location)
        return_value = id_card_reader.get_id_card_info(file_location)
        os.remove(file_location)
        print("Time: ", time.time() - st)
        return JSONResponse(status_code=201, content={"message": return_value})
    except:
        return JSONResponse(status_code=500, content={"message": "An error occurred"})


@app.post("/dkx")
async def dkx(file: UploadFile = File(...)):
    try:
        st = time.time()
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return_value = vehicle_registration_reader.get_vehicle_registration_info(file_location)
        os.remove(file_location)
        print("Time: ", time.time() - st)
        return JSONResponse(status_code=201, content={"message": return_value})
    except:
        return JSONResponse(status_code=500, content={"message": "An error occurred"})


if __name__ == '__main__':
    uvicorn.run('main_fastapi:app', host='0.0.0.0', port=1234)
