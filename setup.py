
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='aimodelshare', #TODO:update
    version='0.0.107',        #TODO:update
    author="Michael Parrott",
    author_email="mikedparrott@modelshare.org",
    description="Deploy locally saved machine learning models to a live rest API and web-dashboard.  Share it with the world via modelshare.org",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.modelshare.org",
    packages=setuptools.find_packages(),
    install_requires=["boto3==1.26.69", "botocore==1.29.82","scikit-learn==1.2.1","onnx>=1.13.1","onnxconverter-common>=1.7.0",
    "regex", "keras2onnx>=1.7.0","tensorflow>=2.12","tf2onnx","skl2onnx>=1.14.0","onnxruntime>=1.7.0","torch>=1.8.1","pydot==1.3.0",
    "importlib-resources==5.10.0","onnxmltools>=1.6.1","Pympler==0.9","docker==5.0.0","wget==3.2","PyJWT>=2.4.0","seaborn>=0.11.2",
    "astunparse==1.6.3","shortuuid>=1.0.8","psutil>=5.9.1","pathlib>=1.0.1","scipy==1.7.0", "protobuf>=3.20.1", "dill", "IPython>=8.12", "scikeras"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    #package_data={'': ['placeholders/model.onnx', 'placeholders/preprocessor.zip']},
    )     
  
