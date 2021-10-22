## Approximation of classic algorithms with neural networks

To clone this repository on your local computer, run in terminal:
```
git clone https://github.com/OlgaChaganova/approxnn
```

To install all dependencies, run:
```
pip install -r requirements.txt
```

To download data for edge detection model, run:

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -qq 'tiny-imagenet-200.zip
```

To download data for binarization model, run (you have to add ```kaggle.json``` file to the working directory before downloading):

```
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d mariajosecastrobleda/noisyoffice
unzip noisyoffice.zip
```

To run the experiment:
```
python main.py <filter_name>
````
where ```<filter_name>``` can be *canny*, *niblack* or *harris*.

You can change hyperparameters written at the file ```config.py```