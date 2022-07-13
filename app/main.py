# import requirements needed
from flask import Flask, render_template, request
from utils import get_base_url
from math import expm1
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 1000
base_url = get_base_url(port)

pricemodel = keras.models.load_model("price_prediction_model.h5")
scaler = MinMaxScaler()




# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# set up the routes and logic for the webserver
@app.route(f'{base_url}' )
def home():
    return render_template('index.html')

@app.route(f'{base_url}/index' )
def index():
    return render_template('index.html')

@app.route(f'{base_url}/products')
def products():
    return render_template('products.html')

@app.route(f'{base_url}/model')
def model():
    return render_template('model.html')

@app.route(f'{base_url}/store')
def store():
    return render_template('store.html')

@app.route(f'{base_url}/store', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(flask.render_template('index.html', prediction_text = ""))
    if request.method == 'POST':
        inp_features = [float(x) for x in request.form.values()]
        
        input_variables = np.array(inp_features)
        
        data= scaler.fit_transform([input_variables])
        
        prediction = pricemodel.predict(data)[0]
        
        output = '{0:.2f}'.format(prediction[0])
        
        return render_template('store.html',prediction_text=output)



# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc7.ai-camp.dev'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
