__author__ = 'alain'

# TODO: CONVERT ALL STRINGS TO CONSTANTS

import pandas as pd
import numpy as np
import madden


class SeasonClassifier(object):
    """
    class: SeasonClassifier

    wrapper object for running ML classifier for a particular season and training data set
    """

    def __init__(self, datafile, reference_data, classifier=madden.DEFAULT_SCIKIT_CLASSIFIER):
        '''
        :param datafile: location of nflLines data file
        :param reference_data: ReferenceData object with lookups
        :return:
        '''
        self.datafile = datafile
        self.reference_data = reference_data
        self.classifier = classifier
        self.features = None

    def readData(self, seasons, dataType):
        '''
        Synopsis: reads data file, computes season records and other fields necessary for classifier

        :param seasons: array of int, which seasons to read
        :param dataType: 'test' or 'train' to specify use
        :return:
        '''
        # read datafile
        dfAllGames = madden.readGamesAll(self.datafile, seasons)
        # compute season records from scores
        dfAllTeams = madden.seasonRecord(dfAllGames, self.reference_data)
        # apply season records and compute other fields for all games
        dfAllGames = madden.processGames(dfAllGames, dfAllTeams, self.reference_data)
        # remove extra year of data
        dfAllGames = dfAllGames[dfAllGames.season.isin(seasons)]

        if dataType == 'train':
            self.trainSeasons = seasons
            self.trainGames = dfAllGames
        elif dataType == 'test':
            self.testSeasons = seasons
            self.testGames = dfAllGames


    def runClassifier(self, yClassifier='favoredWin'):
        """
        Synopsis: Runs the current classifier

        :param yClassifier: column name with the output

        :return:
        """
        self.yClassifier = yClassifier
        self.classifier = madden.runScikitClassifier(self.trainGames, self.features, self.classifier, yClassifier)

    def predict(self, weeks='all'):
        """
        Use the current classifier to predict a given season

        :param weeks: weeks to predict. Usually numeric. If you pass "all", it will try to predict all weeks.

        :return:

        """
        # apply results of classifier to the test set
        dfPredict = madden.predictGames(self.testGames, self.classifier, self.features, self.yClassifier)
        self.predictGames = dfPredict

    def predictAccuracy(self, dataType='test'):
        """
        Prints out the accuracy of this method
        """
        if dataType == 'train':
            dfAllGames = self.trainGames
        elif dataType == 'test':
            dfAllGames = self.testGames

        sc = madden.predictAccuracy(dfAllGames,self.classifier,self.features,self.yClassifier)
        print ("%s predict accuracy = " % dataType, sc)

    def rank(self):
        """
        Apply ranking logic and determine scoring outcomes for league
        """
        dfAll = madden.rankGames(self.predictGames,self.reference_data,self.testSeasons[0])
        self.rankGames = dfAll

    def getWinningScore(self):
        """
        Calculates the winning score of a given season after prediction
        """
        # get winning score for season
        try:
            winningScore = self.reference_data.getSeasonWinner(self.testSeasons[0])
        except:
            # pick lineScore as benchmark if not available - for season in progress
            winningScore = self.rankGames.groupby('season')['lineScore'].sum().values[0]

        #print (winningScore, type(winningScore))
        self.winningScore = winningScore
        return winningScore

    def seasonSummary(self):
        """
        """
        # get full season scores in pandas.Series
        scoreCols = ['lineScore', 'probaScore1','probaScore2','probaScore3']

        winningScore = self.getWinningScore()
        sSeason = self.rankGames.groupby('season')[scoreCols].sum() - winningScore

        # extra info
        sSeason['trainYears'] = str(self.trainSeasons)
        sSeason['classifierType'] = type(self.classifier)

        return sSeason

    def predictSummary(self, week, guessCol = 'probaGuess'):
        """
        Calculates the prediction summary
        """
        dispCols = ['season','gameWeek','Visitor','visitorRecord','Home Team','homeRecord',
            'Line','prevFavoredRecord','prevUnderdogRecord','predict_proba',
            'lineGuess','probaGuess', 'probaAbsGuess', 'predictTeam']

        dfAll = self.rankGames
        dfAll = dfAll[ dfAll.gameWeek == week]

        # rank method 2
        dfAll['predictTeam'] = np.where((dfAll['predict_proba'] - .5) > 0 , dfAll['favorite'], dfAll['underdog'])
        predictCols = ['predictTeam', 'predict_proba', guessCol, 'favorite', 'Line']

        return dfAll[predictCols].sort(guessCol, ascending=False)


class SeasonClassifierCollection(object):
    """
    This class is an attempt to try multiple classifiers
    """
    def __init__(self):
        """
        Constructor. Initializes dictionary of classifiers
        """
        self.classifiersList = dict()

    def addSeasonClassifier(self, oSeason, desc):
        """
        Add one classifier to the list
        """
        self.classifiersList[desc] = oSeason

    def getSeasonClassifier(self,desc):

        return self.classifiersList[desc]

    def listCollection(self):
        """
        """
        listItems = list()
        for k,v in self.classifiersList.items():
            listItems.append(k)

        return listItems

    def summary(self):
        """
        Returns a summary of the results
        """
        dfSummary = pd.DataFrame()

        for k,v in self.classifiersList.items():
            seasonSummary = v.seasonSummary()
            if dfSummary is None:
                dfSummary = seasonSummary
            else:
                dfSummary = dfSummary.append(seasonSummary)


        return (dfSummary.sort_index())

