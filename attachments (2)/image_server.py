import torch
import numpy as np
from PIL import Image
import urllib
import json
import pickle

from flask import Flask, request, jsonify

app = Flask(__name__)
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

with open("map.json", 'r') as f:
    idx_class = json.load(f)

@app.route('/', methods=['GET'])
def search():
    url = request.args.get('url', None)
    filename = request.args.get('file', None)
    if url:
        filename = get_img_filename(url)
    label = get_results(filename)
    label= idx_class[label]
    return label

def get_img_filename(url):
    filename = url.split('/')[-1]
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    return filename

def get_results(filename):
    
    #input_image = Image.open(filename)
    inp=[19,162.59,52,34.4,86]
    i=np.asarray(inp)
    x=i.reshape(1,-1)
    a = pickle_model.predict(x)
    #predicted_label = idx_class[a]
    return a[0]

if __name__ == "__main__":
    #app.secret_key = "shdjehdie3u92edhw2"
    app.run(debug=True,host='0.0.0.0',port=9008)

#app.secret_key = "shdjehdie3u92edhw2"
#app.run(debug=True, host='0.0.0.0')
