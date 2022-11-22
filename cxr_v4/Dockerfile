FROM python:3.7-slim
# Update
RUN apt-get update -y
RUN apt-get install gcc -y
RUN apt-get install libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev -y

WORKDIR /ai
COPY . .
RUN pip install -r requirements.txt


EXPOSE 8004

CMD ["python", "main.py"]