FROM nvidia/cuda:10.0-cudnn7-runtime
RUN apt update
RUN apt -y install python3 
RUN apt -y install python3-pip
RUN mkdir /opt/rubiks_cube

RUN pip3 install numpy==1.16.4
RUN pip3 install Keras==2.2.4
RUN pip3 install tqdm==4.23.2
RUN pip3 install pycuber==0.2.2
RUN pip3 install tensorflow-gpu
RUN pip3 install h5py==2.10.0

COPY . /opt/rubiks_cube
WORKDIR /opt/rubiks_cube
RUN mv pycuber/util.py /usr/local/lib/python3.6/dist-packages/pycuber/util.py

ENTRYPOINT ["python3",  "TrainCubeNN.py"]
