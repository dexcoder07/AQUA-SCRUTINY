from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired, ValidationError


class InputForm(FlaskForm):

    ph = IntegerField(label='Ph Value', validators=[DataRequired()])
    hardness = IntegerField(label='Hardness', validators=[DataRequired()])
    dis_solids = IntegerField(label='Dissolved solids', validators=[DataRequired()])
    Chloramines = IntegerField(label='Chloramines', validators=[DataRequired()])
    Sulfate = IntegerField(label='Sulfate', validators=[DataRequired()])
    Conductivity = IntegerField(label='Conductivity', validators=[DataRequired()])
    org_carbon = IntegerField('Organic Carbon', validators=[DataRequired()])
    Trihalomethanes = IntegerField(label='Trihalomethanes', validators=[DataRequired()])
    Turbidity = IntegerField(label='Turbidity', validators=[DataRequired()])
    submit = SubmitField(label='Predict')


