
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
    install_requires=["boto3==1.21.20", "botocore==1.24.20","scikit-learn==0.24.2","onnx==1.12.0","onnxconverter-common>=1.7.0", "regex",
    "keras2onnx>=1.7.0","tensorflow==2.9.2","tf2onnx","skl2onnx>=1.8.0","onnxruntime>=1.7.0","torch>=1.8.1","pydot==1.3.0","importlib-resources==5.10.0",
      "onnxmltools>=1.6.1","Pympler==0.9","docker==5.0.0","wget==3.2","PyJWT>=2.4.0","seaborn>=0.11.2","astunparse==1.6.3","shortuuid>=1.0.8","psutil>=5.9.1","pathlib>=1.0.1"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True)     
  
