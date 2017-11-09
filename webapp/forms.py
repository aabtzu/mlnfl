from wtforms import Form, IntegerField, validators
import datetime

TODAY = datetime.date.today()
CURRENT_SEASON = TODAY.year

class ActionForm(Form):
    season = IntegerField('season', [validators.required(), validators.NumberRange(min=2014, max=CURRENT_SEASON)], default=CURRENT_SEASON)
    week = IntegerField('week', [validators.required(), validators.NumberRange(min=1, max=17)], default=1)



