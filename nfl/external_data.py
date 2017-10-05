__author__ = 'amit'

import pandas
import bs4
import requests
import urllib2
import os
import datetime
import argparse

CURRENT_SEASON = 2017
SPREADS_URL = 'http://www.footballlocks.com/nfl_point_spreads.shtml'
SCORES_URL = 'http://www.pro-football-reference.com/years/%d/games.htm' % CURRENT_SEASON

CODE_DIR = "".join([os.environ['MLNFL_ROOT'], os.path.sep])
path_to_lines = CODE_DIR + "data/lines/"
lines_file = path_to_lines + "nflAllLines.csv"


def read_lines():
    # read in the master lines file
    df_lines = pandas.read_csv(lines_file)
    df_lines.Date = pandas.to_datetime(df_lines.Date)
    return df_lines


def save_lines(df_lines):
    '''
    Synopsis: save the master lines file

    :param df_lines: pandas DataFrame of all lines and scores

    :return: None
    '''
    df_lines.to_csv(lines_file, index=False)


def scrape_spreads():

    r = requests.get(SPREADS_URL)
    data = r.text

    soup = bs4.BeautifulSoup(data, "lxml")

    # get the tables w/ the spreads
    tt = soup.findAll("table", {"width": "580"})

    df_spreads = pandas.DataFrame()
    for i in range(2):  # hard coded table number most of the time
    #for i in range(1):  # hard coded table number for last week of season
        dfs = pandas.read_html(str(tt[i]), )
        df_spreads = df_spreads.append(dfs[0][range(4)])

    df_spreads.index = range(len(df_spreads))
    df_spreads.columns = ['date', 'favorite', 'spread', 'underdog']

    filter_bad = df_spreads.favorite == 'Favorite'
    df_spreads = df_spreads[~filter_bad]

    # get the home favorite
    df_spreads['home_favorite'] = (df_spreads.favorite.str.contains('^At ')) | (df_spreads.favorite.str.contains('\(At '))
    # fix any spreads that are tied (PK)
    df_spreads.loc[df_spreads.spread.astype(unicode).str.contains('PK'), 'spread'] = -.1 # need -1 for some reason

    # flip sign on spread for away favorite
    df_spreads['factor'] = 1
    df_spreads.loc[df_spreads.home_favorite == True,'factor'] = -1
    df_spreads['spreads2'] = df_spreads.spread.astype(float) * df_spreads.factor

    # get the home team
    df_spreads['home_team'] = df_spreads.favorite
    home_filter = df_spreads.underdog.str.contains('^At ') | df_spreads.underdog.str.contains('\(at ')
    df_spreads.loc[home_filter, 'home_team'] = df_spreads.loc[home_filter, 'underdog']
    df_spreads.home_team = df_spreads.home_team.str.replace('^At ', '')
    df_spreads.home_team = df_spreads.home_team.str.replace('\(At .*\)', '')
    df_spreads.home_team = df_spreads.home_team.str.replace('\(.*\)', '')
    df_spreads['datetime'] = pandas.to_datetime('2017/'+df_spreads.date.str.split(" ", expand=True)[0],
                                                format='%Y/%m/%d').dt.date

    return df_spreads

def merge_spreads(df_spreads, df_lines):

    # find the right week/game and update the spread
    week_filter = (df_lines.Date <= df_spreads.datetime.max()) & (df_lines.Date >= df_spreads.datetime.min())

    for ii, rr in df_spreads.iterrows():
        print ii, rr['home_team'], rr['spreads2']
        if ('NY' in rr['home_team']) | ('LA' in rr['home_team']):
            rr['home_team'] = rr['home_team'].split(' ')[1]
        game_filter = df_lines[week_filter]['Home Team'].str.contains(rr['home_team'])
        irow = df_lines[week_filter][game_filter].index[0]
        df_lines.loc[irow, 'Line'] = rr['spreads2']

    return df_lines


def scrape_scores(week, season=CURRENT_SEASON):

    scores_url = SCORES_URL
    r  = requests.get(scores_url)
    data = r.text

    soup = bs4.BeautifulSoup(data, 'lxml')

    # get the tables w/ the spreads
    tt = soup.findAll("table")

    df_scores = pandas.read_html(str(tt[0]), )[0]

    # Make sure to enter the correct week number
    week_filter = df_scores.Week == str(week)
    df_week = df_scores[week_filter]

    # rename home game col
    old_home_col = df_week.columns[5]
    home_col = 'home_game'
    df_week.rename(columns={old_home_col: home_col}, inplace=True)

    winner_col = 'Winner/tie'
    loser_col = 'Loser/tie'
    winner_pts_col = 'PtsW'
    loser_pts_col = 'PtsL'

    # means that winner was away team, loser was home team
    away_filter = df_week[home_col] == '@'

    # populate these new cols
    new_cols = ['home_pts', 'away_pts', 'home_team', 'away_team']
    for cc in new_cols:
        df_week[cc] = None

    df_week.loc[~away_filter, 'home_team']  = df_week.loc[~away_filter, winner_col]
    df_week.loc[away_filter, 'home_team']  = df_week.loc[away_filter, loser_col]
    df_week.loc[~away_filter, 'away_team']  = df_week.loc[~away_filter, loser_col]
    df_week.loc[away_filter, 'away_team']  = df_week.loc[away_filter, winner_col]

    df_week.loc[~away_filter, 'home_pts']  = df_week.loc[~away_filter, winner_pts_col]
    df_week.loc[away_filter, 'home_pts']  = df_week.loc[away_filter, loser_pts_col]
    df_week.loc[~away_filter, 'away_pts']  = df_week.loc[~away_filter, loser_pts_col]
    df_week.loc[away_filter, 'away_pts']  = df_week.loc[away_filter, winner_pts_col]

    return df_week

def merge_scores(df_week, week, season, df_lines):

    # find the right week/game and update the score
    week_filter = (df_lines.season == season) & (df_lines.week == week)

    for ii, rr in df_week.iterrows():
        print ii, rr['home_team'], rr['home_pts']
        game_filter = df_lines[week_filter]['Home Team'].str.contains(rr['home_team'])
        irow = df_lines[week_filter][game_filter].index[0]
        print df_lines.irow(irow)['Home Team']
        df_lines.loc[irow, 'Home Score'] = rr['home_pts']
        df_lines.loc[irow, 'Visitor Score'] = rr['away_pts']

    return df_lines


def get_current_week(df_lines, current_season=CURRENT_SEASON):

    today = datetime.datetime.today().date()
    date_filter = (df_lines.Date > today) & (df_lines.season == current_season)
    current_week = df_lines[date_filter].week.min()
    return int(current_week)


def verify_data(df_data, data_type):
    print "verifying %s data:" % data_type
    print df_data
    ans = raw_input("accept (y/n): ")
    if ans.lower() == 'y':
        return True
    return False


if __name__ == "__main__":

    # read lines file and get current week
    df_lines = read_lines()
    season = CURRENT_SEASON
    current_week = get_current_week(df_lines, season)

    # define input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--week', '-w', action='store', default=current_week, dest='game_week',
                    type=int, help='Pass the week number to make the picks')
    parser.add_argument('--scores', action='store_true', dest='scores')
    parser.add_argument('--spreads', action='store_true', dest='spreads')
    args = parser.parse_args()

    # get and save scores
    if args.scores:
        week = args.game_week

        print "getting scores of week %d of %d season ..." % (week, season)
        df_week = scrape_scores(week, season)
        df_lines = merge_scores(df_week, week, season, df_lines)
        if verify_data(df_week, 'scores'):
            save_lines(df_lines)

    if args.spreads:
        # get and save spreads
        print "getting most recent spreads ..."
        df_spreads = scrape_spreads()
        df_lines = merge_spreads(df_spreads, df_lines)
        if verify_data(df_spreads, 'spreads'):
            save_lines(df_lines)

