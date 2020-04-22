from __future__ import absolute_import, division, print_function, unicode_literals
import os
# 默认显示所有等级
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
# 只显示warning和error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
# Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory

import logging
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

import datetime

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)
KEYSPACE = "mykeyspace2"

new_model = keras.models.load_model('my_model.h5')
def ImageToMatrix(filename):
    im = Image.open(filename)
    im = im.resize((28, 28))
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.reshape(data,(height,width))
    data=255-data  #反色
    data = data / 255.0
    return data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def createKeySpace():
    cluster = Cluster(contact_points=['172.18.0.2'], port=9042)
    session = cluster.connect()
    #session.execute('drop keyspace mykeyspace1;')
    log.info("Creating keyspace...")
    try:
        session.execute("""
           CREATE KEYSPACE %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
           """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)
        log.info("creating table...")
        session.execute("""
           CREATE TABLE mytable (
               time varchar,
               name varchar,
               value varchar,
               PRIMARY KEY (time, name)
           )
           """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)
    cluster.shutdown()

createKeySpace();

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            data = ImageToMatrix(file)
            plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
            probability_model = tf.keras.Sequential([new_model, tf.keras.layers.Softmax()])
            data = (np.expand_dims(data, 0))
            predictions_single = probability_model.predict(data)

            def insertTable():
                cluster = Cluster(contact_points=['172.18.0.2'], port=9042)
                session = cluster.connect()
                session.set_keyspace(KEYSPACE)
                start = datetime.datetime.now()
                session.execute('insert into mytable (time, name, value) '
                                'values (%s, %s, %s);',
                                [str(start), filename, class_names[np.argmax(predictions_single)]])
                rows = session.execute("select * from mytable;")
                for row in rows:
                    print(row)
            insertTable();
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Welcome to the Picture Recognition System</h1>
    <form method=post enctype=multipart/form-data>
    <p><input type=file name=file>
       <input type=submit value=Upload>
    </form>
    '''

@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')



