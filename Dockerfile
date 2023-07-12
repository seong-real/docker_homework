FROM tensorflow/tensorflow:latest
LABEL maintainer "wjstjdfuf <wjsjdfuf98@naver.com>"
WORKDIR /root
RUN apt-get update
RUN apt-get install -y python3-pip
COPY requirments.txt requirments.txt
RUN pip3 install -r requirments.txt
ENTRYPOINT ["python3"]
CMD ["model.py"]
