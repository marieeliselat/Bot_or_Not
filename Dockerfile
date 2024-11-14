FROM python:3

RUN pip install requests
RUN pip install pydantic
RUN pip install textblob
RUN pip install numpy
RUN pip install scikit-learn

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
