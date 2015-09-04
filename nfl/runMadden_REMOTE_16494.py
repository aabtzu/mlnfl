__author__ = 'amit.bhattacharyya'

import madden
import numpy as np



def runSeasonRolling(trainYears, testYear, olookups, trainFreq = 1):

     # training data set - includes one extra year for prev yr record
    seasons = np.array(trainYears)

    # use different test set
    seasonTest = np.array(testYear) # should be only one year

    # read the games
    dfAllGames = madden.readGamesAll(madden.PATH_TO_NFL_LINES, seasons)
    dfGamesTest = madden.readGamesAll(madden.PATH_TO_NFL_LINES, seasonTest)

    # compile season record for all teams
    dfAllTeams = madden.seasonRecord(dfAllGames, olookups)
    dfTestTeams = madden.seasonRecord(dfGamesTest, olookups)

    # apply season records and compute other fields for all games
    dfAllGames = madden.processGames(dfAllGames, dfAllTeams, olookups)
    dfGamesTest = madden.processGames(dfGamesTest, dfTestTeams, olookups)

    # remove extra year of data
    dfAllGames = dfAllGames[dfAllGames.season.isin(seasons)]
    dfGamesTest = dfGamesTest[dfGamesTest.season.isin(seasonTest)]

    # define independent variables for logistic regression
    features = ['favoredRecord','underdogRecord',  # current year records of both teams
                'prevFavoredRecord','prevUnderdogRecord', # prev year records, helps early in season when only few games played
                'gameWeek',  # week in season, should make a good/bad record later in season more important
                'absLine',  # absolute value of spread since favored team already determined
                'divisionGame', # T/F, usually more competitive rivalry games, i.e. bad teams still win home division games.
                'favoredHomeGame', # T/F, important since output of classifier is "did the favored team win?"
                ]


    nweeks = 17
    dfWeeks = None
    trainWeeks = 1
    for i in range(nweeks):
        iw = i + 1 # actual week of season

        print "season %d, week %d" % (seasonTest[0],iw)


        if trainFreq == 0:
            dfGames = dfAllGames
            dfTest = dfGamesTest[dfGamesTest.gameWeek == iw]
        else:
            if (iw % trainFreq) == 0:
                trainWeeks = iw

            dfGames = dfAllGames.append(dfGamesTest[dfGamesTest.gameWeek < trainWeeks])
            dfTest = dfGamesTest[dfGamesTest.gameWeek == iw]

        # run the logistic regression
        logreg = madden.runML(dfGames,features)

        # apply results of logistic regression to the test set
        dfPredict = madden.predictGames(dfTest,logreg,features)

        # apply ranking logic and determine scoring outcomes for league
        dfAll = madden.rankGames(dfPredict,olookups,seasonTest[0])

        g = dfAll.groupby('gameWeek',as_index=False)['lineScore','probaScore1','probaScore2','probaScore3'].sum()
        g.index = [iw]
        g['season'] = seasonTest[0]
        #print(g)

        if dfWeeks is None:
            dfWeeks = g
        else:
            dfWeeks = dfWeeks.append(g)

        g = dfWeeks.groupby('season',as_index=False)['lineScore','probaScore1','probaScore2','probaScore3'].sum()
        g.index = [seasonTest[0]]
        g['train'] = seasons
        g['trainFreq'] = trainFreq
    return (g)

def runSeasonLoop(trainStart, trainLen, classifier, path_to_lines, reference_data):

    dfLoop = None
    testYear = 0

    while testYear < 2014:
        # determine test and train years
        testYear = trainStart + trainLen
        trainYears = range(testYear-trainLen,testYear)


        print testYear, trainYears

        # define test and train years
        seasons = np.array(trainYears)
        seasonTest = np.array([testYear]) # should be only one year

        # read the games
        dfAllGames = madden.readGamesAll(path_to_lines, seasons)
        dfGamesTest = madden.readGamesAll(path_to_lines, seasonTest)

        # compile season record for all teams
        dfAllTeams = madden.seasonRecord(dfAllGames, reference_data)
        dfTestTeams = madden.seasonRecord(dfGamesTest, reference_data)

        # apply season records and compute other fields for all games
        dfAllGames = madden.processGames(dfAllGames, dfAllTeams, reference_data)
        dfGamesTest = madden.processGames(dfGamesTest, dfTestTeams, reference_data)

        # remove extra year of data
        dfAllGames = dfAllGames[dfAllGames.season.isin(seasons)]
        dfGamesTest = dfGamesTest[dfGamesTest.season.isin(seasonTest)]

        # define independent variables for logistic regression
        features = ['favoredRecord','underdogRecord',  # current year records of both teams
                'prevFavoredRecord','prevUnderdogRecord', # prev year records, helps early in season when only few games played
                'gameWeek',  # week in season, should make a good/bad record later in season more important
                'absLine',  # absolute value of spread since favored team already determined
                'divisionGame', # T/F, usually more competitive rivalry games, i.e. bad teams still win home division games.
                'favoredHomeGame', # T/F, important since output of classifier is "did the favored team win?"
                ]

        # run classifier
        classifier = madden.runScikitClassifier(dfAllGames,features,classifier)
        # apply results of logistic regression to the test set
        dfPredict = madden.predictGames(dfGamesTest,classifier,features)
        # apply ranking logic and determine scoring outcomes for league
        dfAll = madden.rankGames(dfPredict,reference_data,seasonTest[0])

        # get winning score for season
        try:
            winningScore = reference_data.getSeasonWinner(testYear)
        except:
            winningScore = dfAll.groupby('season')['lineScore'].sum().values[0]

        print winningScore, type(winningScore)

        # get full season scores in pandas.Series
        scoreCols = ['lineScore', 'probaScore1','probaScore2','probaScore3',]
        sSeason = dfAll.groupby('season')[scoreCols].sum() - winningScore

        # extra info
        sSeason['trainYears'] = str(seasons)
        sSeason['classifierType'] = type(classifier)
        sSeason['classifier'] = classifier



        if dfLoop is None:
            dfLoop = sSeason
        else:
            dfLoop = dfLoop.append(sSeason)


        trainStart += 1


    return dfLoop