FROM ubuntu:20.04

# Set the working directory
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update -y && apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick python3-pip python3-dev build-essential ffmpeg unzip git wget curl

RUN pip install -r requirements.txt

RUN cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml

COPY . /app

ENV FFMPEG_BINARY=/bin/ffmpeg 

RUN mkdir -p /user/share/fonts/googlefonts
RUN unzip -d /usr/share/fonts/googlefonts ./Montserrat.zip
RUN fc-cache -fv

RUN python3 ./src/load_model.py

CMD ["/bin/bash"]