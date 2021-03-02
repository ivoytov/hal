FROM python:3.8
WORKDIR /hal
RUN apt-get -y update && apt-get install -y libhdf5-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
CMD [ "python", "./momentum.py", "fetch", "--host", "10.108.0.2", "--port","4002"]
