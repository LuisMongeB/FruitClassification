FROM python:3.10.12

# python installs
COPY requirements_azureml.txt .
RUN pip install -r requirements_azureml.txt && rm requirements_azureml.txt
WORKDIR /fruitclassification2/
# set command
CMD ["bash"]