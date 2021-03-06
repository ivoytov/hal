FROM python:3.8-slim
WORKDIR /hal
RUN apt-get -y update && apt-get install -y libhdf5-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
CMD [ "python", "./trader.py", "--host", "10.108.0.2", "--port","4001", "--store", "/hdf/data/bond_data.h5", "--model", "/hdf/bond_model_210225.joblib", "--log", "/hdf/hal.log"]
EXPOSE 5555/tcp
