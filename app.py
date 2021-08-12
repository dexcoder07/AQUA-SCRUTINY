from flask import Flask, render_template, jsonify, request
import numpy as np
from forms import *
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ea856da3d403d75563d1cd43'


@app.route("/", methods=['GET', 'POST'])
def screen():
    lay = InputForm()
    if lay.validate_on_submit():
        print(lay.ph.data)
        """
                params = [lay.ph.data, lay.hardness, lay.dis_solids.data, lay.Chloramines.data,
                  lay.Sulfate.data, lay.Conductivity.data, lay.org_carbon.data, lay.Trihalomethanes.data,
                  lay.Turbidity.data]
        f_params = np.array([params])
        print(f_params)
        """
    return render_template('base.html', form=lay)


if __name__ == '__main__':
    app.run(debug=True)
