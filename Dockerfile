FROM python:3.8-slim
WORKDIR /hal
RUN apt-get -y update && apt-get install -y libhdf5-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
EXPOSE 5555/tcp
