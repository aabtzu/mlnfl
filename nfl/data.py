import pandas
import dateutil.parser as dp
import datetime
import numpy as np
from sklearn import linear_model

import lookup

MAX_WEEK = 17


def getWeek(seasonStart, gameDateStr):
    """
    :Synopsis: determine week of season

    :param seasonStart: datetime.date for start of season
    :param gameDateStr: date str for date game was played
    :returns: week of the season that game was played
    """

    gameDate = dp.parse(gameDateStr).date()
    week = int(np.ceil((gameDate - seasonStart).days / 7.0)) + 1
    return week


def sameDivision(team1, team2, olookups):
    """
    :Synopsis: determine if two teams are in the same division

    :param team1: team1
    :param team2: team1
    :param olookups: Lookup object with a "teams" dictionary
    :returns: True if the teams are in the same division
    """

    conf1 = olookups.lookupCSV('teams', team1, 'league')
    conf2 = olookups.lookupCSV('teams', team2, 'league')

    div1 = olookups.lookupCSV('teams', team1, 'division')
    div2 = olookups.lookupCSV('teams', team2, 'division')

    if conf1 == conf2 and div1 == div2:
        return True
    else:
        return False


def readGamesSingleSeason(dataRoot, season):
    """
    :Synopsis: Read a csv file of game scores and spreads into a pandas data frame

    :param dataRoot: Path to the root of the data directory
    :param season: Season as an integer
    :returns: A pandas.DataFrame with all the data
    """

    dataFile = dataRoot + "lines/nfl%slines.csv" % str(season)
    dfAllGames = pandas.read_csv(dataFile)
    return dfAllGames


def readGamesAll(dataRoot, seasons):
    """
    :Synopsis: Read a csv file of game scores and spreads into a pandas data frame

    :param dataRoot: Path to the root of the data directory
    :param seasons: list of integers of which seasons to import
    :returns: A pandas.DataFrame with all the data, adds one previous season in addition to
    """

    dataFile = dataRoot + "lines/nflAllLines.csv"
    dfAllGames = pandas.read_csv(dataFile)
    # need one extra season for prev year records
    seasons2 = np.insert(seasons, 0, seasons.min() - 1)

    dfAllGames = dfAllGames[dfAllGames.season.isin(seasons2)]
    return dfAllGames


def seasonRecord(dfAllGames, olookups):
    """
    :Synopsis: compile season stats by team

    :param dfAllGames: pandas.DataFrame with list of all games played
    :param: olookups: Lookup object with a "teams" dictionary
    :returns: pandas.DataFrame with games played, games won, lost, record to date, home/away, division game by team
    """

    # list of seasons/teams to loop over
    seasons = dfAllGames.season.unique()
    teams = dfAllGames.Visitor.unique()
    dfAllSeasons = None

    # loop over seasons
    for ss in seasons:
        dfSeason = dfAllGames[dfAllGames.season == ss]
        dfAllTeams = None
        # loop over teams
        for ii, tt in enumerate(teams):
            # print ii,tt

            dfTeam = dfSeason[(dfSeason.Visitor == tt) | (dfSeason['Home Team'] == tt)]
            dfTeam['gamesPlayed'] = range(1, len(dfTeam.index) + 1)  # index 1 thur 16
            dfTeam['team'] = tt
            dfTeam['homeGame'] = dfSeason['Home Team'] == tt  # true for home game
            dfTeam['wonGame'] = ((dfTeam['Visitor Score'] < dfTeam['Home Score']) & dfTeam['homeGame']) | (
                (dfTeam['Visitor Score'] > dfTeam['Home Score']) & (dfTeam['homeGame'] == False)) # did team win
            dfTeam['gamesWon'] = dfTeam['wonGame'].cumsum()  # cumulative games won
            dfTeam['homeGamesWon'] = (dfTeam['wonGame'] & dfTeam['homeGame']).cumsum()  # cumulative home games won
            dfTeam['gamesLost'] = dfTeam['gamesPlayed'] - dfTeam['gamesWon']  # cumulative games lost
            dfTeam['winPct'] = dfTeam['gamesWon'] / dfTeam['gamesPlayed'] # winning pct by week
            dfTeam['homeGamesPlayed'] = dfTeam['homeGame'].cumsum()  # cumulative home games played
            dfTeam['homeWinPct'] = dfTeam['homeGamesWon'] / dfTeam['homeGamesPlayed'] # home winning pct by week

            # determine if division game
            opponent = list()
            divGame = list()
            for ii, row in dfTeam.iterrows():
                if row['Home Team'] == row['team']:
                    team2 = row['Visitor']
                else:
                    team2 = row['Home Team']

                opponent.append(team2)
                divGame.append(sameDivision(row['team'].lower(), team2.lower(), olookups))

            dfTeam['opponent'] = opponent
            dfTeam['divGame'] = divGame

            dfAllTeams = pandas.concat([dfAllTeams, dfTeam])
        dfAllSeasons = pandas.concat([dfAllSeasons, dfAllTeams])
    return dfAllSeasons


def getSeasonStart(olookups, season):
    """
    :Synopsis: determine starting date of a given season which is defined as first monday night game

    :param olookups: Lookup object with a "seasons" dictionary
    :param season: int of season to lookup
    :returns: datetime.date of first monday night game of season
    """

    seasonStr = olookups.lookupCSV('seasons', str(season), 'start')
    seasonStart = dp.parse(seasonStr).date()
    return seasonStart


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


def processGames(dfAllGames, dfAllTeams, olookups):
    """
    :Synopsis:  apply season record and other stats to all games played

    :param dfAllGames: pandas.DataFrame of each game to be included in training set
    :param dfAllTeams: pandas.DataFrame of teams and records for all the seasons
    :param olookups: Lookup object with a "seasons" and "teams" dictionary

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

    seasons = dfAllGames.season.unique()  # get list of seasons

    # loop over each game and apply season record to date
    for gg in dfAllGames.iterrows():
        game = gg[1]  # need this because of how the generator works ??

        # get season info for this game
        season = game['season']
        seasonStart = getSeasonStart(olookups, season)
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
        homeWin = int(int(game['Home Score']) > int(game['Visitor Score'])) # 0/1 did home team win ?
        scoreDiff = int(game['Home Score'] - game['Visitor Score']) # difference in score
        favoredWin = int((game['Line'] * scoreDiff) > 0) # 0/1 did favored team win = sign of (line * score diff)
        divGame = int(sameDivision(game['Home Team'].lower(), game['Visitor'].lower(), olookups))  # 0/1 division game
        favoredHomeGame = int(game['Line'] > 0) # 0/1 is the home team favored

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
    dfAllGames['favoredHomeGame'] = favoredHome
    dfAllGames['divisionGame'] = division
    dfAllGames['homeWin'] = winner
    dfAllGames['favoredWin'] = favored
    dfAllGames['gameWeek'] = gameNum

    dfAllGames['homeRecord'] = homePct
    dfAllGames['visitorRecord'] = visitorPct
    dfAllGames['favoredRecord'] = favRecord
    dfAllGames['underdogRecord'] = dogRecord
    dfAllGames['prevFavoredRecord'] = prevFavRecord
    dfAllGames['prevUnderdogRecord'] = prevDogRecord

    dfAllGames['absLine'] = abs(dfAllGames['Line']) # need abs to rank easily

    return dfAllGames


def getTrainData(dfAllGames, featuresList, yClassify='favoredWin',maxTrainWeek=17):
    """
    :Synopsis: extract features and classifiers for sklearn routines

    :param dfAllGames: pandas.DataFrame with training data and outcomes for all games
    :param featuresList: list of columns to include in training
    :param yClassify: (optional) column of the classifier to predict, default = "favoredWin"
    :param maxTrainWeek: (optional) number of weeks of each season to use for training, default = 17

    n.b. the default goal of the ML training is to answer "did the favored team win ?"
    n.b. set maxTrainWeek<17 if you dont want to use all weeks of the season.
    last week of season is often meaningless since some teams rest players once playoff seeds are determined.

    :returns: X,y and arrays for use in sklearn routines
    """

    dfTrain = dfAllGames[dfAllGames.gameWeek <= maxTrainWeek]
    y = dfTrain[yClassify].tolist()
    X = dfTrain[featuresList].as_matrix()

    return X, y


def runLogisticRegression(X, y):
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
    print "training data accuracy = ", sc

    return logreg


def runML(dfAllGames, featuresList, yClassifier="favoredWin"):
    """
    :Synopsis: run machine learning on training data

    :param dfAllGames: pandas.DataFrame with training data and outcomes for all games
    :param featuresList: list of columns to include in training
    :param yClassifier: column of the classifier to predict

    :returns: fitted linear_model.LogisticRegression object
    """

    train_X, train_y = getTrainData(dfAllGames, featuresList)
    logreg = runLogisticRegression(train_X, train_y)
    return logreg


def predictGames(dfAllGames, logreg, featuresList):
    """
    :Synopsis: apply results of logistic regression to test data

    :param dfAllGames: pandas.DataFrame with training data and outcomes for all games
    :param logreg: fitted linear_model.LogisticRegression object
    :param featuresList: list of columns used in training/test data

    :returns: augmented pandas.DataFrame with predictions of games based on logistic regression results
    """

    dfPredict = dfAllGames
    predict_X, predict_y = getTrainData(dfAllGames, featuresList)

    # proba_predict gives the probability that the favored team wins
    # 1 = 100% chance favored team wins
    # 0 = 100% change underdog win

    # proba_predict_abs takes distance away the absolute value from 50%,
    # this allows us to rank teams based on "who will win" not "how likely is favored team to win"

    # get results from pre-computed logreg object
    p = logreg.predict(predict_X)  # 0/1 classifier
    pp = logreg.predict_proba(predict_X)  # probability of favored team winning
    dfxn = logreg.decision_function(predict_X)  # not sure how this is helpful yet

    dfPredict['predict_proba'] = pp[:, 1]
    dfPredict['predict_proba_abs'] = abs(pp[:, 1] - 0.5)
    dfPredict['decision_fxn'] = dfxn

    scoreDiff = dfPredict['Home Score'] - dfPredict['Visitor Score']
    # predicted win = if the team that is predicted to win by logreg actually wins
    #                 not necessarily the favored team
    dfPredict['predictWin'] = 1 * ((scoreDiff * dfPredict['Line'] * (dfPredict['predict_proba'] - 0.5) ) > 0)
    return dfPredict


def rankGames(dfPredict, olookups, season):
    """
    :Synopsis: determine weekly rankings for games based on predict from logistic regression

    :param dfPredict: pandas.DataFrame with predicted winners
    :param olookups: Lookup object with a "seasons" dictionary
    :param season: int of season

    :returns: pandas.DataFrame with results of ranked games and computed scores from outcomes
    """

    # the actual winner of the league historically
    winningScore = int(olookups.lookupCSV('seasons', str(season), 'winner'))

    weeks = dfPredict.gameWeek.unique()
    dfAll = None

    for ww in weeks:
        iw = dfPredict[dfPredict.gameWeek == ww].index
        dfWeek = dfPredict.loc[iw]
        nw = len(iw)

        # name of Home Team is arbitrary tie breaker for same spread - but at least it is reproducible
        dfLine = dfWeek.sort(['absLine', 'Home Team'])

        # determine guess
        # several possibilities for predicting by probability
        # 1. always pick favored team, rank by probability of win
        # 2. pick winner based on abs(probability - .5), rank by probability
        # 3. pick winner based on abs(probability - .5), rank by abs(probability - .5)
        dfLine['lineGuess'] = dfLine['absLine'].rank('first') + (16 - nw)
        dfLine['probaGuess'] = dfLine['predict_proba'].rank('first') + (16 - nw)
        dfLine['probaAbsGuess'] = dfLine['predict_proba_abs'].rank('first') + (16 - nw)

        # determine score of pick by spread
        dfLine['lineScore'] = dfLine['lineGuess'] * dfLine['favoredWin']

        # determine score of pick by probability
        dfLine['probaScore1'] = dfLine['probaGuess'] * dfLine['favoredWin']
        dfLine['probaScore2'] = dfLine['probaGuess'] * dfLine['predictWin']
        dfLine['probaScore3'] = dfLine['probaAbsGuess'] * dfLine['predictWin']

        if ww == 1:
            dfAll = dfLine
        else:
            dfAll = dfAll.append(dfLine)

    # aggregate results by week
    g = dfAll.groupby('gameWeek')['lineScore', 'probaScore1', 'probaScore2', 'probaScore3'].sum()

    # print out summary table with final scores and how they compare to previous winners
    dd = [g.sum(), g.sum() - winningScore]
    ss = pandas.DataFrame(dd).transpose()
    ss[2] = ss[1] > 0
    ss.columns = ['score', 'win by', 'win']
    print ss

    return dfAll

