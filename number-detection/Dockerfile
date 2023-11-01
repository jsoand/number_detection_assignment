FROM python:3.10.12-slim

WORKDIR /code

#RUN pip install tensorflow numpy mnist keras fastapi "uvicorn[standard]" Pillow python-multipart
COPY ./requirements.txt /code/requirements.txt

RUN pip install -r requirements.txt

COPY ./app /code

EXPOSE 5000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]