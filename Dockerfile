# creates virtual ubuntu in docker image
FROM ubuntu:18.04

# maintainer of docker file
MAINTAINER David Bouget <david.bouget@sintef.no>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# installing python3
RUN apt-get update -y && \
    apt-get install python3-pip -y && \
    apt-get -y install sudo && \
    apt-get update && \
    pip3 install bs4 && \
    pip3 install requests && \
    apt-get install python3-lxml -y && \
    pip3 install Pillow && \
    apt-get install libopenjp2-7 -y && \
    apt-get install libtiff5 -y

# install curl
RUN apt-get install curl -y

# install nano
RUN apt-get install nano -y

# install git (OBS: using -y is conveniently to automatically answer yes to all the questions)
RUN apt-get update && apt-get install -y git

# give user sudo access and access to python directories
RUN useradd -m ubuntu && echo "ubuntu:ubuntu" | chpasswd && adduser ubuntu sudo
ENV PYTHON_DIR /usr/bin/python3
RUN chown ubuntu $PYTHON_DIR -R
USER ubuntu

# Python
RUN pip3 install tensorflow==1.14.0
RUN pip3 install tensorflow-gpu==1.14.0
RUN pip3 install progressbar2
RUN pip3 install nibabel
RUN pip3 install h5py==2.10.0
RUN pip3 install scipy
RUN pip3 install scikit-image==0.16.2
RUN pip3 install progressbar2
RUN pip3 install tqdm
RUN pip3 install SimpleITK==1.2.4
RUN pip3 install numpy==1.19.3

RUN mkdir /home/ubuntu/src
WORKDIR "/home/ubuntu/src"
COPY src/ $WORKDIR
WORKDIR "/home/ubuntu"
COPY Dockerfile $WORKDIR
COPY main.py $WORKDIR

RUN mkdir /home/ubuntu/resources
USER root
RUN chown -R ubuntu:ubuntu /home/ubuntu/resources
RUN chmod -R 777 /home/ubuntu/resources
USER ubuntu
EXPOSE 8888

#RUN echo 'alias python=python3' >> ~/.bashrc

# CMD ["/bin/bash"]
ENTRYPOINT ["python3","/home/ubuntu/main.py"]




