FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip3 install scipy
RUN pip3 install torchvision==0.2.2
RUN pip3 install scikit-learn
RUN pip3 install matplotlib
RUN pip3 install visdom
RUN pip3 install torchmetrics
RUN pip3 install kornia