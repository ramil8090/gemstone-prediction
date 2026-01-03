FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime keras-image-helper==0.0.2

ARG MODEL_NAME=gemstone_classifier_resnet101.onnx
ENV MODEL_NAME=${MODEL_NAME}

ARG MODEL_DATA=gemstone_classifier_resnet101.onnx.data

COPY ${MODEL_NAME} ${MODEL_NAME}
COPY ${MODEL_DATA} ${MODEL_DATA}

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]