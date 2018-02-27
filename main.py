from flask import Flask



app = Flask(__name__)

from get_data import *
from predict import *
from train import *



if __name__ == '__main__':
	app.run(debug=True)
