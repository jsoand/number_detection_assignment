FROM python:3.10.12-slim

WORKDIR /code

RUN pip install fastapi "uvicorn[standard]"

COPY ./app /code

EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]