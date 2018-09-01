
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request
from flask import render_template
import forms
import os

import nfl_picks2 as picks
import external_data as ed

MLNFL_ROOT_DIR = picks.MLNFL_ROOT_DIR
PICKS_DIR = "".join([MLNFL_ROOT_DIR, os.path.sep, 'picks'])

ACTION_DICT = {
    'Scrape spreads': {'action':'spreads', 'fxn':ed.scrape_spreads },
    'Scrape scores': {'action':'scores', 'fxn':ed.scrape_scores },
    'Go Bananas': {'action':'picks', 'fxn':picks.main },
}

#df_week = scrape_scores(week, season)

app = Flask(__name__)

@app.route('/test', methods=('GET', 'POST'))
def test():
    form = forms.ActionForm()
    return render_template("base.html",  form=form)

@app.route('/', methods=('GET', 'POST'))
def main():

    if (request.method == 'POST'):
        form = forms.ActionForm(request.form)
        submit = request.form.get('submit')
        action = ACTION_DICT[submit]['action']
        action_fxn = ACTION_DICT[submit]['fxn']

        if form.validate():
            week = int(request.form.get("week").lower())
            season = int(request.form.get("season").lower())

            if action == 'spreads':
                df = ed.scrape_spreads()
            if action == 'scores':
                df = ed.scrape_scores(week, season)
            if action == 'picks':
                df = picks.main(season, week, PICKS_DIR)
            return df.to_html()


            # return request.method + " >> " +  "%s %d %d" % (submit, week, season)

    else:
        form = forms.ActionForm()
    return render_template("main.html",  form=form)

