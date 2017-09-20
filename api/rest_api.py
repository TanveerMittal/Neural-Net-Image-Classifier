#!flask/bin/python
from flask import Flask, jsonify, request
import os
app = Flask(__name__)

json = [
    {
        'id': u'Dictionary',
        'title': u'datatype',
        'description': u'Get request'
    }
]

@app.route('/api/GET/JSON', methods=['GET'])
def get_tasks():
    return jsonify(json)

@app.route('/api/POST/JSON', methods=['POST'])
def get_tasks():
    x = request.form(["JSON"])
    print x

if __name__ == '__main__':
    app.run(debug=True)