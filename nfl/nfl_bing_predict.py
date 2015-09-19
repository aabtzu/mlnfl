#!/bin/python

__author__ = 'alain ledon'

"""
Raw script to get NFL Bing predict results.
Required packages:
- requests
- pandas
- bs4
"""

import requests
import pandas as pd
import bs4

BING_PREDICT_NFL_URL = r"http://www.bing.com/search?q=nfl+predictions"


def run():
    '''
    Goes to Bing Predict and fetches current week predictions
    It doesn't work properly

    :return:
    '''
    response = requests.get(BING_PREDICT_NFL_URL)
    soup = bs4.BeautifulSoup(response.text)

    bing_results = soup.select("p")
    #bing_results = soup.select("div.meg_container.p")

    results = [unicode(x.string).encode('utf8', 'replace') for x in bing_results]
    print results
    home_teams = []
    visitor_teams = []
    favorite_teams = []
    probability_win = []
    for i in xrange(0, len(results), 3):
        if results[i].find("at"):
            if results[i] == 'None':
                break

        home, visitor = results[i].split(' at ')
        prediction = results[i + 2].split(' win (')
        favorite = prediction[0]
        probability = float(prediction[1].split('%')[0])
        home_teams.append(home)
        visitor_teams.append(visitor)
        favorite_teams.append(favorite)
        probability_win.append(probability)

    df_bing = pd.DataFrame({'Home': home_teams, 'Visitor': visitor_teams,
                            'Favorite': favorite_teams, 'Probability': probability_win})

    print df_bing.sort(columns='Probability', ascending=False)

if __name__ == "__main__":
    run()