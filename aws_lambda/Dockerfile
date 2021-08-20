# Official AWS image
FROM public.ecr.aws/lambda/python:3.8

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the relevant directories and files
COPY handwritten_multi_digit_number_recognition/ ./handwritten_multi_digit_number_recognition
COPY aws_lambda/app.py ./app.py
COPY artifacts ./artifacts

CMD ["app.lambda_handler"]