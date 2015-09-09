from __future__ import division
from __future__ import print_function

__author__ = 'alainledon'

WEEK_TO_PICK = 1

import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
pd.set_option('expand_frame_repr', False)

import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

import madden

from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble

from referencedata import ReferenceData

# define the root directory for the nfl code in $MLNLF_ROOT
codeDir = "".join([os.environ['MLNFL_ROOT'], os.path.sep])
dataRoot = "".join([codeDir, "data", os.path.sep])

os.chdir(codeDir)

logging.info("Base directory = {0}".format(codeDir))
logging.info("Data directory = {0}".format(dataRoot))

# location of lookup files

lookupFiles = {'teams' : {'file': 'nflTeams.csv'}, 'seasons': {'file': 'seasons.csv'}, }

lookupDir = "".join([dataRoot, 'lookup', os.path.sep])

logging.info("lookupFiles = %s" % lookupFiles)
logging.info("lookupDir = %s" % lookupDir)

# load reference data
reference_data = ReferenceData(lookupDir)

# train on previous 3 yrs of data
testYear = [2015]
trainYears = range(testYear[0]-3,testYear[0])

# training data set - includes one extra year for prev yr record
seasons = np.array(trainYears)
logging.info("training seasons >> {0}".format(seasons))

# get training data
# 1 - read all the games
path_to_lines = dataRoot + "lines/"
dfAllGames = madden.readGamesAll(path_to_lines, seasons)
# 2 - compile season record for all teams
dfAllTeams = madden.seasonRecord(dfAllGames, reference_data)
# 3 - apply season records and compute other fields for all games
dfAllGames = madden.processGames(dfAllGames, dfAllTeams, reference_data)
# 4 - remove extra year of data
dfAllGames = dfAllGames[dfAllGames.season.isin(seasons)]

# use different test set
season_test = np.array(testYear) # should be only one year
logging.info("results for >> {0}".format(season_test))
# 1 - read all the games
dfGamesTest = madden.readGamesAll(path_to_lines, season_test)
# 2 - compile season record for all teams
dfTeamsTest = madden.seasonRecord(dfGamesTest,reference_data)
# 3 - apply season records and compute other fields for all games
dfGamesTest = madden.processGames(dfGamesTest, dfTeamsTest, reference_data)
# 4 - remove extra year of data
dfGamesTest = dfGamesTest[dfGamesTest.season.isin(season_test)]

# define independent variables for logistic regression
features = ['favoredRecord','underdogRecord',  # current year records of both teams
            'prevFavoredRecord','prevUnderdogRecord', # prev year records, helps early in season when only few games played
            'gameWeek',  # week in season, should make a good/bad record later in season more important
            'absLine',  # absolute value of spread since favored team already determined
            'divisionGame', # T/F, usually more competitive rivalry games, i.e. bad teams still win home division games.
            'favoredHomeGame', # T/F, important since output of classifier is "did the favored team win?"
            ]

# run the classifer
random_state = 11
svm_classifier = svm.SVC(kernel='poly', probability=True, random_state=random_state)
lr_classifier = linear_model.LogisticRegression(C=1e5)

svm_trained_classifier = madden.runScikitClassifier(dfAllGames, madden.FEATURE_COLUMNS, svm_classifier)
lr_trained_classifier = madden.runScikitClassifier(dfAllGames, madden.FEATURE_COLUMNS, lr_classifier)

# predict one week of current season
week_number = WEEK_TO_PICK

# should be only one year
season_test = np.array(testYear)

dfGamesTest = madden.readGamesAll(path_to_lines, season_test)
dfTeamsTest = madden.seasonRecord(dfGamesTest,reference_data)
dfGamesTest = madden.processGames(dfGamesTest, dfTeamsTest, reference_data)
dfGamesTest = dfGamesTest[dfGamesTest.season.isin(season_test)]

# pick only this weeks games for predict
dfTest = dfGamesTest[dfGamesTest.gameWeek == week_number]

###################################################################################################################
# apply results of logistic regression to the test set
df_svm_predict = madden.predictGames(dfTest, svm_trained_classifier, features)
# apply ranking logic and determine scoring outcomes for league
dfAll = madden.rankGames(df_svm_predict, reference_data, season_test[0])

# display weekly ranking output

# ranking methods choices
# 0. pick based on spread
# 1. always pick favored team, rank by probability of win
# 2. pick winner based on abs(probability - .5), rank by probability
# 3. pick winner based on abs(probability - .5), rank by abs(probability - .5)

dispCols = ['season','gameWeek','Visitor','visitorRecord','Home Team','homeRecord',
            'Line','prevFavoredRecord','prevUnderdogRecord','predict_proba',
            'lineGuess','probaGuess', 'probaAbsGuess', 'predictTeam']


dfAll['predictTeam'] = np.where((dfAll['predict_proba'] - .5) > 0 , dfAll['favorite'], dfAll['underdog'])
guessCol = 'probaGuess'
predictCols = ['gameWeek','predictTeam', 'predict_proba', guessCol, 'favorite','lineGuess', 'Line']

logging.info("Picks for week {0} using SVM".format(week_number))
print("\nPicks using SVM")
print(dfAll[predictCols].sort(guessCol, ascending=False))

week_filter = dfAll.gameWeek == week_number
print("\nPicks using SVM")
print(dfAll[week_filter][predictCols].sort(guessCol, ascending=False))
###################################################################################################################

###################################################################################################################
# apply results of logistic regression to the test set
df_lr_predict = madden.predictGames(dfTest, lr_trained_classifier, features)
# apply ranking logic and determine scoring outcomes for league
dfAll = madden.rankGames(df_lr_predict, reference_data, season_test[0])

# Use Method 2
dfAll['predictTeam'] = np.where((dfAll['predict_proba'] - .5) > 0 , dfAll['favorite'], dfAll['underdog'])

logging.info("Picks for week {0} using LogReg".format(week_number))
print("\nPicks using LogReg")
print(dfAll[predictCols].sort(guessCol, ascending=False))

week_filter = dfAll.gameWeek == week_number
print("\nPicks using LogReg")
print(dfAll[week_filter][predictCols].sort(guessCol, ascending=False))
###################################################################################################################

# display weekly ranking output for spread method

# ranking methods choices
# 0. pick based on spread
# 1. always pick favored team, rank by probability of win
# 2. pick winner based on abs(probability - .5), rank by probability
# 3. pick winner based on abs(probability - .5), rank by abs(probability - .5)

predictCols = ['favorite','lineGuess', 'absLine', 'Line', 'favoredHomeGame', 'divisionGame', 'favoredRecord']

sortCols = ['absLine','favoredHomeGame', 'divisionGame', 'favoredRecord', 'favorite']
dfSpread = dfAll[predictCols].sort(sortCols , ascending=False)
#print(dfSpread.to_csv(sys.stdout, sep=',', index=False))
print("\nHere are the Spread picks:")
print(dfSpread)