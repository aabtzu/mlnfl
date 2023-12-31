#!/bin/python

# TODO: CONVERT ALL STRINGS TO CONSTANTS

import pandas as pd
import dateutil.parser as dp
import numpy as np

from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing

from .referencedata import ReferenceData

import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

MAX_WEEK = 17

FILENAME_ALL_LINES = "nflAllLines.csv"
LOG_REG_CLASSIFIER = linear_model.LogisticRegression(C=1e5)
RANDOM_STATE = 11
SVM_CLASSIFIER = svm.SVC(kernel='poly', probability=True, random_state=RANDOM_STATE)

DEFAULT_SCIKIT_CLASSIFIER = LOG_REG_CLASSIFIER

FEATURE_COLUMNS = ['favoredRecord',
                   'underdogRecord',  # Current year records of both teams
                   'prevFavoredRecord',
                   'prevUnderdogRecord',  # Prev year records, helps early in season when only few games played
                   'gameWeek',  # Week in season, should make a good/bad record later in season more important
                   'absLine',  # absolute value of spread since favored team already determined
                   'divisionGame',  # T/F, More competitive rivalry games, i.e. bad teams still win home division games
                   'favoredHomeGame'  # T/F, important since output of classifier is "did the favored team win?"
                   ]

def getWeek(seasonStartDate, gameDateStr):
    """
    :Synopsis: determine week of season

    :param seasonStartDate: datetime.date for start of season
    :param gameDateStr: date str for date game was played
    :returns: week of the season that game was played
    """

    gameDate = dp.parse(gameDateStr).date()
    week = int(np.ceil((gameDate - seasonStartDate).days / 7.0)) + 1
    return week


def sameDivision(team1, team2, ref_data):
    """
    :Synopsis: determine if two teams are in the same division

    :param team1: team1
    :param team2: team1
    :param ref_data: Lookup object with a "teams" dictionary
    :returns: True if the teams are in the same division
    """

    conf1 = ref_data.get_team_league(team1)
    conf2 = ref_data.get_team_league(team2)

    div1 = ref_data.get_team_division(team1)
    div2 = ref_data.get_team_division(team2)

    if conf1 == conf2 and div1 == div2:
        return True
    else:
        return False


def readGamesSingleSeason(dataRootDir, season):
    """
    :Synopsis: Read a csv file of game scores and spreads into a pandas data frame

    :param dataRootDir: Path to the root of the data directory
    :param season: Season as an integer
    :returns: A pandas.DataFrame with all the data
    """

    #TODO: Fix hardcoded path
    dataFile = "".join([dataRootDir, "nfl{0}lines.csv".format(season)])
    all_games_df = pd.read_csv(dataFile)
    return all_games_df


def readGamesAll(dataRoot, seasons, filename_all_lines = FILENAME_ALL_LINES):
    """
    :Synopsis: Read a csv file of game scores and spreads into a pandas data frame

    :param dataRoot: Path to the root of the data directory
    :param seasons: list of integers of which seasons to import
    :returns: A pandas.DataFrame with all the data, adds one previous season in addition to
    """

    dataFile = "".join([dataRoot, filename_all_lines])
    all_games_df = pd.read_csv(dataFile)
    # need one extra season for prev year records
    seasons2 = np.insert(seasons, 0, seasons.min() - 1)

    all_games_df = all_games_df[all_games_df.season.isin(seasons2)]
    return all_games_df


def seasonRecord(all_games_df, refdata):
    """
    :Synopsis: compile season stats by team

    :param all_games_df: pandas.DataFrame with list of all games played
    :param: refdata: Lookup object with a "teams" dictionary
    :returns: pandas.DataFrame with games played, games won, lost, record to date, home/away, division game by team
    """

    # list of seasons/teams to loop over
    seasons = all_games_df.season.unique()
    teams = all_games_df.Visitor.unique()
    all_seasons_df = None

    # loop over seasons
    for season in seasons:
        season_df = all_games_df[all_games_df.season == season]
        all_teams_df = None
        # loop over teams
        for ii, team in enumerate(teams):
            # print("%d - %s" % (ii, team))
            team_df = season_df[(season_df.Visitor == team) | (season_df['Home Team'] == team)]

            team_df['gamesPlayed'] = range(1, len(team_df.index) + 1)  # index 1 thur 16
            team_df['team'] = team
            team_df['homeGame'] = season_df['Home Team'] == team  # true for home game
            team_df['wonGame'] = (
                                (team_df['Visitor Score'] < team_df['Home Score']) & team_df['homeGame']) | (
                                (team_df['Visitor Score'] > team_df['Home Score']) & (team_df['homeGame'] == False)
                                ) # did team win
            team_df['gamesWon'] = team_df['wonGame'].cumsum()  # cumulative games won
            team_df['homeGamesWon'] = (team_df['wonGame'] & team_df['homeGame']).cumsum()  # cumulative home games won
            team_df['gamesLost'] = team_df['gamesPlayed'] - team_df['gamesWon']  # cumulative games lost
            team_df['winPct'] = team_df['gamesWon'] / team_df['gamesPlayed'] # winning pct by week
            team_df['homeGamesPlayed'] = team_df['homeGame'].cumsum()  # cumulative home games played
            team_df['homeWinPct'] = team_df['homeGamesWon'] / team_df['homeGamesPlayed'] # home winning pct by week

            # determine if division game
            opponent = list()
            divGame = list()
            for ii, row in team_df.iterrows():
                if row['Home Team'] == row['team']:
                    team2 = row['Visitor']
                else:
                    team2 = row['Home Team']

                opponent.append(team2)
                divGame.append(sameDivision(row['team'], team2, refdata))

            team_df.loc[:, 'opponent'] = opponent
            team_df.loc[:, 'divGame'] = divGame

            all_teams_df = pd.concat([all_teams_df, team_df])
        all_seasons_df = pd.concat([all_seasons_df, all_teams_df])
    return all_seasons_df


def getRecord(dfTeams, season, team, week):
    """
    :Synopsis: get record of team by week in season
    n.b. the week in season is NOT the same as the number of games played because of bye weeks

    :param dfTeams: pandas.DataFrame with compiled season record by week
    :param season: int of season to lookup
    :param team: string to team to lookup
    :param week: int of week of in season
    :returns: float of winning percentage for the week of the season
    """

    dfSeason = dfTeams[dfTeams.season == season]
    record = float(dfSeason[(dfSeason['team'] == team) & (dfSeason['gamesPlayed'] == week - 1)]['winPct'])
    return record


def processGames(all_games_df, dfAllTeams, reference_data):
    """
    :Synopsis:  apply season record and other stats to all games played

    :param all_games_df: pandas.DataFrame of each game to be included in training set
    :param dfAllTeams: pandas.DataFrame of teams and records for all the seasons
    :param reference_data: Lookup object with a "seasons" and "teams" dictionary

    :returns: augmented pandas.DataFrame and includes additional variables necessary for ML training.
    """

    # init
    favoredHome = list()
    homePct = list()
    visitorPct = list()
    gameNum = list()
    winner = list()
    favored = list()
    division = list()
    favRecord = list()
    dogRecord = list()
    prevFavRecord = list()
    prevDogRecord = list()
    favorite = list()
    underdog = list()

    seasons = all_games_df.season.unique()  # get list of seasons

    # loop over each game and apply season record to date
    # pandas data frame iterrows returns (index, Series) tuple.
    for i, game in all_games_df.iterrows():
        # get season info for this game
        season = game['season']
        seasonStart = reference_data.getSeasonStartDate(season)
        prevSeason = season - 1

        # dates and week of game
        gameDateStr = game['Date']
        gameWeek = getWeek(seasonStart, gameDateStr)

        # get cumulative record up to previous week
        if gameWeek > 1:
            homeRecord = getRecord(dfAllTeams, season, game['Home Team'], gameWeek)
            visitorRecord = getRecord(dfAllTeams, season, game['Visitor'], gameWeek)

            # positive spread if home team is favored
            if game['Line'] > 0:
                favoredRecord = homeRecord
                underdogRecord = visitorRecord
            else:
                favoredRecord = visitorRecord
                underdogRecord = homeRecord
        else:
            # for first week of season
            homeRecord = 0.0
            visitorRecord = 0.0
            favoredRecord = 0.0
            underdogRecord = 0.0

        # score, win and line info for each game
        # cant expect this to work if game has not been played yetx
        try:
            homeWin = int(int(game['Home Score']) > int(game['Visitor Score'])) # 0/1 did home team win ?
            scoreDiff = int(game['Home Score'] - game['Visitor Score']) # difference in score
            favoredWin = int((game['Line'] * scoreDiff) > 0) # 0/1 did favored team win = sign of (line * score diff)
        except:
            homeWin = np.NaN
            favoredWin = np.NaN
            scoreDiff = np.NaN

        divGame = int(sameDivision(game['Home Team'], game['Visitor'], reference_data))  # 0/1 division game
        favoredHomeGame = int(game['Line'] > 0) # 0/1 is the home team favored
        if favoredHomeGame:
            favoredTeam = game['Home Team']
            underdogTeam = game['Visitor']
        else:
            favoredTeam = game['Visitor']
            underdogTeam = game['Home Team']

        # get record from previous season
        if prevSeason in seasons:
            prevHomeRecord = getRecord(dfAllTeams, prevSeason, game['Home Team'], MAX_WEEK)
            prevVisitorRecord = getRecord(dfAllTeams, prevSeason, game['Visitor'], MAX_WEEK)
            if game['Line'] > 0:
                prevFavoredRecord = prevHomeRecord
                prevUnderdogRecord = prevVisitorRecord
            else:
                prevFavoredRecord = prevVisitorRecord
                prevUnderdogRecord = prevHomeRecord
        else:
            prevFavoredRecord = None
            prevUnderdogRecord = None

        # append list of each game -- need to think of vectorized way to do this
        favorite.append(favoredTeam)
        underdog.append(underdogTeam)
        favoredHome.append(favoredHomeGame)
        homePct.append(homeRecord)
        visitorPct.append(visitorRecord)
        gameNum.append(gameWeek)
        winner.append(homeWin)
        favored.append(favoredWin)
        division.append(divGame)
        favRecord.append(favoredRecord)
        dogRecord.append(underdogRecord)
        prevFavRecord.append(prevFavoredRecord)
        prevDogRecord.append(prevUnderdogRecord)

    # fill in data frame with all new columns -- need to think of vectorized way to do this
    all_games_df['favorite'] = favorite
    all_games_df['underdog'] = underdog

    all_games_df['favoredHomeGame'] = favoredHome
    all_games_df['divisionGame'] = division
    all_games_df['homeWin'] = winner
    all_games_df['favoredWin'] = favored
    all_games_df['gameWeek'] = gameNum

    all_games_df['homeRecord'] = homePct
    all_games_df['visitorRecord'] = visitorPct
    all_games_df['favoredRecord'] = favRecord
    all_games_df['underdogRecord'] = dogRecord
    all_games_df['prevFavoredRecord'] = prevFavRecord
    all_games_df['prevUnderdogRecord'] = prevDogRecord

    all_games_df['absLine'] = abs(all_games_df['Line']) # need abs to rank easily

    return all_games_df


def getTrainData(all_games_df, featuresList, yClassifier='favoredWin', maxTrainWeek=20):
    """
    :Synopsis: extract features and classifiers for sklearn routines

    :param all_games_df: pandas.DataFrame with training data and outcomes for all games
    :param featuresList: list of columns to include in training
    :param yClassifier: column of the classifier to predict, default = 'favoredWin'
    :param maxTrainWeek: (optional) number of weeks of each season to use for training, default = 17

    n.b. the default goal of the ML training is to answer "did the favored team win ?"
    n.b. set maxTrainWeek<17 if you don't want to use all weeks of the season.
    last week of season is often meaningless since some teams rest players once playoff seeds are determined.

    :returns: X,y and arrays for use in sklearn routines
    """

    df_train = all_games_df[(all_games_df.gameWeek <= maxTrainWeek)]
    y = df_train[yClassifier].tolist()
    X = df_train[featuresList].values

    return X, y


def runScikitLogisticRegression(X, y):
    """
    :Synopsis: get logistic regression object for features/classifiers

    :param X: matrix of features for logistic regression
    :param y: list of classifiers for logistic regression

    :returns: fitted linear_model.LogisticRegression object
    """

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, y)

    # compute training accuracy
    sc = logreg.score(X, y)
    logging.info("training data accuracy = {0}".format(sc))

    return logreg


def runGraphLabClassifier(all_games_sf, features, yClassifier, gl_classifier):
    """
    :Synopsis: run GraphLab classifier on training data
    :param all_games_sf: graphlab.SFrame with training data and outcomes for all games
    :return: fitted graphlab model olbject
    """

    gl_model = gl_classifier.create(all_games_sf, yClassifier, features=features)
    return gl_model


def runScikitClassifier(all_games_df, features_list, classifier=DEFAULT_SCIKIT_CLASSIFIER, yClassifier ='favoredWin'):
    """
    :Synopsis: run machine learning on training data

    :param all_games_df: pandas.DataFrame with training data and outcomes for all games
    :param features_list: list of columns to include in training
    :param classifier: which classifier to use, default = linear_model.LogisticRegression()
    :param yClassifier: the variable to predict

    :returns: fitted model object based on input classifier
    """

    X, y = getTrainData(all_games_df, features_list, yClassifier)

    # try normalizing
    #X = preprocessing.normalize(X)
    #y = preprocessing.normalize(y)
    classifier.fit(X, y)

    # compute training accuracy
    sc = classifier.score(X, y)
    logging.info("training data accuracy = {0}".format(sc))
    return classifier


def predictGames(all_games_df, classifier, featuresList, yClassifier = 'favoredWin'):
    """
    :Synopsis: apply results of logistic regression to test data

    :param all_games_df: pandas.DataFrame with training data and outcomes for all games
    :param classifier: fitted linear_model.LogisticRegression object
    :param featuresList: list of columns used in training/test data

    :returns: augmented pandas.DataFrame with predictions of games based on logistic regression results
    """

    df_predict = all_games_df # [all_games_df[yClassifier].notnull()]
    predict_X, predict_y = getTrainData(df_predict, featuresList, yClassifier)

    # proba_predict gives the probability that the favored team wins
    # 1 = 100% chance favored team wins
    # 0 = 100% change underdog win

    # proba_predict_abs takes distance away the absolute value from 50%,
    # this allows us to rank teams based on "who will win" not "how likely is favored team to win"

    # get results from pre-computed classifier object
    p = classifier.predict(predict_X)  # 0/1 classifier
    pp = classifier.predict_proba(predict_X)  # probability of favored team winning
    #dfxn = classifier.decision_function(predict_X)  # not sure how this is helpful yet

    df_predict['predict_proba'] = pp[:, 1]
    df_predict['predict_proba_abs'] = abs(pp[:, 1] - 0.5)
    #df_svm_predict['decision_fxn'] = dfxn

    score_diff = df_predict['Home Score'] - df_predict['Visitor Score']
    # predicted win = if the team that is predicted to win by classifier actually wins
    #                 not necessarily the favored team
    df_predict['predictWin'] = 1 * ((score_diff * df_predict['Line'] * (df_predict['predict_proba'] - 0.5)) > 0)
    return df_predict


def predictAccuracy(all_games_df, classifier, featuresList, yClassifier):
    """
    Synopsis: compute predict accuracy using classifier results
    :param all_games_df: pandas.DataFrame with training/test data and outcomes
    :param classifier: fitter classifier object
    :param featuresList: list of columns used in training data
    :param yClassifier: name of the column that is classified
    :return:
    """

    dfPredict = all_games_df[all_games_df[yClassifier].notnull()]
    predict_X, predict_y = getTrainData(dfPredict, featuresList, yClassifier)

    # example of conditional statement within list comprehension
    # [x+1 if x >= 45 else x+5 for x in l]

    predict_y_int = [ int(yy) if pd.notnull(yy) else yy for yy in predict_y]
    sc = classifier.score(predict_X, predict_y_int)

    return sc


def rankGames(dfPredict, reference_data, season):
    """
    :Synopsis: determine weekly rankings for games based on predict from logistic regression

    :param dfPredict: pandas.DataFrame with predicted winners
    :param reference_data: Lookup object with a "seasons" dictionary
    :param season: int of season

    :returns: pandas.DataFrame with results of ranked games and computed scores from outcomes
    """

    # the actual winner of the league historically
    try:
        winningScore = reference_data.getSeasonWinner(season)
    except:
        winningScore = 0

    weeks = dfPredict.gameWeek.unique()
    dfAll = None

    for ww in weeks:
        # get an index to all the games on the given week
        iw = dfPredict[dfPredict.gameWeek == ww].index
        dfWeek = dfPredict.loc[iw]
        nw = len(iw)

        # name of Home Team is arbitrary last tie breaker  - but at least it is reproducible
        sortCols = ['absLine', 'favoredHomeGame', 'divisionGame', 'favoredRecord', 'Home Team']
        dfLine = dfWeek.sort_values(sortCols)

        # determine guess
        # several possibilities for predicting by probability
        # 1. always pick favored team, rank by probability of win
        # 2. pick winner based on abs(probability - .5), rank by probability
        # 3. pick winner based on abs(probability - .5), rank by abs(probability - .5)
        dfLine['lineGuess'] = dfLine['absLine'].rank(method='first') + (16 - nw)
        dfLine['probaGuess'] = dfLine['predict_proba'].rank(method='first') + (16 - nw)
        dfLine['probaAbsGuess'] = dfLine['predict_proba_abs'].rank(method='first') + (16 - nw)

        # determine score of pick by spread
        dfLine['lineScore'] = dfLine['lineGuess'] * dfLine['favoredWin']

        # determine score of pick by probability
        dfLine['probaScore1'] = dfLine['probaGuess'] * dfLine['favoredWin']
        dfLine['probaScore2'] = dfLine['probaGuess'] * dfLine['predictWin']
        dfLine['probaScore3'] = dfLine['probaAbsGuess'] * dfLine['predictWin']

        if dfAll is None:
            dfAll = dfLine
        else:
            dfAll = dfAll.append(dfLine)

    # aggregate results by week
    #g = df_all_picks.groupby('gameWeek')['lineScore', 'probaScore1', 'probaScore2', 'probaScore3'].sum()

    # print out summary table with final scores and how they compare to previous winners
    #dd = [g.sum(), g.sum() - winningScore]
    #ss = pd.DataFrame(dd).transpose()
    #ss[2] = ss[1] > 0
    #ss.columns = ['score', 'win by', 'win']
    #print(ss)

    return dfAll




