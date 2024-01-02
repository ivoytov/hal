FROM python:3.8-slim
WORKDIR /hal
RUN apt-get -y update && apt-get install -y libhdf5-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
EXPOSE 5555/tcp
CMD [ "python", "./trader.py", "--host", "104.131.36.68", "--port","4002", "--store", "/hdf/data/bond_data.h5", "--model", "/hdf/bond_model_210225.joblib", "--log", "/hdf/hal.log"]
