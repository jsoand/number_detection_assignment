from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from keras import models
import uvicorn
import time
import numpy
from PIL import Image
import os

model = models.load_model('mymodel.keras')
origins= [
    "*"
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convertImage(dst):
    img = numpy.asarray(Image.open(dst).convert('L').resize((28, 28)), float)
    return img

@app.post('/detect')
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        dst = str(time.time())+'.png'
        with open(dst, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    # convert image
    img = convertImage(dst)

    # transform image
    img = numpy.expand_dims(img, axis=0)
    img = (img/255)-0.5
    img = numpy.expand_dims(img, axis=3)

    # predict image
    predictions = model.predict(img)
    label = numpy.argmax(predictions)

    # delete image
    os.remove(dst)
    return str(label)
