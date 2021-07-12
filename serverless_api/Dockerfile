# Official AWS image
FROM public.ecr.aws/lambda/python:3.8

# Install gcc (required for editdistance)
RUN yum -y install gcc-c++

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
COPY handwritten_digit_string_recognition/ ./handwritten_digit_string_recognition
COPY serverless_api/app.py ./app.py
COPY artifacts ./artifacts

CMD ["app.lambda_handler"]