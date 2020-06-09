FROM ubuntu:20.04
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install python3-pip -y
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["09_flask_api.py"]
