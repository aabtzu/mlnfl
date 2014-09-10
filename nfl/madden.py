from __future__ import division
from __future__ import print_function
import pandas
import dateutil.parser as dp
import numpy as np
from sklearn import linear_model
import os

from referencedata import ReferenceData

MAX_WEEK = 17

PATH_TO_NFL_LINES = '/Users/alainledon/gitdev/bitbucket.org/littlea1/mlkaggle/nfl/data/lines/'

FILENAME_ALL_LINES = "nflAllLines.csv"

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
    all_games_df = pandas.read_csv(dataFile)
    return all_games_df


def readGamesAll(dataRoot, seasons, filename_all_lines = FILENAME_ALL_LINES):
    """
    :Synopsis: Read a csv file of game scores and spreads into a pandas data frame

    :param dataRoot: Path to the root of the data directory
    :param seasons: list of integers of which seasons to import
    :returns: A pandas.DataFrame with all the data, adds one previous season in addition to
    """

    dataFile = "".join([dataRoot, filename_all_lines])
    all_games_df = pandas.read_csv(dataFile)
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

            team_df['opponent'] = opponent
            team_df['divGame'] = divGame

            all_teams_df = pandas.concat([all_teams_df, team_df])
        all_seasons_df = pandas.concat([all_seasons_df, all_teams_df])
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


def getTrainData(all_games_df, featuresList, yClassify='favoredWin',maxTrainWeek=17):
    """
    :Synopsis: extract features and classifiers for sklearn routines

    :param all_games_df: pandas.DataFrame with training data and outcomes for all games
    :param featuresList: list of columns to include in training
    :param yClassify: (optional) column of the classifier to predict, default = "favoredWin"
    :param maxTrainWeek: (optional) number of weeks of each season to use for training, default = 17

    n.b. the default goal of the ML training is to answer "did the favored team win ?"
    n.b. set maxTrainWeek<17 if you don't want to use all weeks of the season.
    last week of season is often meaningless since some teams rest players once playoff seeds are determined.

    :returns: X,y and arrays for use in sklearn routines
    """

    dfTrain = all_games_df[all_games_df.gameWeek <= maxTrainWeek]
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
    print ("training data accuracy = ", sc)

    return logreg


def runML(all_games_df, featuresList, yClassifier="favoredWin"):
    """
    :Synopsis: run machine learning on training data

    :param all_games_df: pandas.DataFrame with training data and outcomes for all games
    :param featuresList: list of columns to include in training
    :param yClassifier: column of the classifier to predict

    :returns: fitted linear_model.LogisticRegression object
    """

    train_X, train_y = getTrainData(all_games_df, featuresList)
    logreg = runLogisticRegression(train_X, train_y)
    return logreg


def predictGames(all_games_df, logreg, featuresList):
    """
    :Synopsis: apply results of logistic regression to test data

    :param all_games_df: pandas.DataFrame with training data and outcomes for all games
    :param logreg: fitted linear_model.LogisticRegression object
    :param featuresList: list of columns used in training/test data

    :returns: augmented pandas.DataFrame with predictions of games based on logistic regression results
    """

    dfPredict = all_games_df
    predict_X, predict_y = getTrainData(all_games_df, featuresList)

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
        iw = dfPredict[dfPredict.gameWeek == ww].index
        dfWeek = dfPredict.loc[iw]
        nw = len(iw)

        # name of Home Team is arbitrary last tie breaker  - but at least it is reproducible
        sortCols = sortCols = ['absLine','favoredHomeGame', 'divisionGame', 'favoredRecord', 'Home Team']
        dfLine = dfWeek.sort(sortCols)

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
    print(ss)

    return dfAll

