FROM public.ecr.aws/lambda/python:3.7

COPY file_objects/. ./

RUN pip install -r ./requirements.txt

CMD ["lambda_function.lambda_handler"]
