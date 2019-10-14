from __future__ import print_function
import os
import json
import pickle
import io
import sys
from postprocess import pre_process,FEATURE_COLUMNS
from utils import ScoringService
import numpy as np
import json
import request
import flask
import pandas as pd

preprocessor_path  = "/home/ec2-user/dsp-linear/model/linear-pipeline.pkl"
model_path = "/home/ec2-user/dsp-linear/model/finalized_model.sav"

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model(model_path) is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/raw_requests', methods=['POST'])
def raw_request():
    df = None
    df = flask.request.get_json(force=True)
    data = pd.io.json.json_normalize(df)
    data=data[FEATURE_COLUMNS]
    pre_process(data)
    preprocessor = ScoringService.get_preprocessor(preprocessor_path)
    data = preprocessor.transform(data)
    predictions = ScoringService.predict(data,model_path)
    out = io.StringIO()
    pd.DataFrame({'results':predictions.flatten()}).to_csv(out, index=False,sep=',',header=['score'])
    result = out.getvalue()
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
