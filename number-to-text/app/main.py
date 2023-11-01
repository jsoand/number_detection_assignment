from typing import Annotated
from fastapi import FastAPI, Path, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

number_list = ["zero","one","two","three","four","five","six","seven","eight","nine"]

@app.get('/number/{number}')
def upload(number: Annotated[int, Path(title="The number to convert to word")]):
    if number:
        return number_list[number].capitalize()
    
    raise HTTPException(status_code=400, detail="Missing number query parameter")
