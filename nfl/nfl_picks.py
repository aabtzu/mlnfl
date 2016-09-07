from __future__ import division
from __future__ import print_function

__author__ = 'alainledon'

WEEK_TO_PICK = 1

import os
import numpy as np
import pandas as pd
import sys
import madden
import logging
import argparse

from sklearn import linear_model
from sklearn import svm
from referencedata import ReferenceData

pd.options.mode.chained_assignment = None
pd.set_option('expand_frame_repr', False)

# get the working directory from the environment variable MLNFL_ROOT
MLNFL_ROOT_DIR = os.environ['MLNFL_ROOT']
print(MLNFL_ROOT_DIR)

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--week', '-w', action='store', default=WEEK_TO_PICK, dest='game_week',
                    type=int, help='Pass the week number to make the picks')
parser.add_argument('--directory', '-d', action='store', default="".join([MLNFL_ROOT_DIR, os.path.sep, 'picks']),
                    dest='picks_dir', help='Pass the target directory to get a csv with the picks')
args = parser.parse_args()

# predict one week of current season
week_number = args.game_week

# define the root directory for the nfl code in $MLNLF_ROOT
codeDir = "".join([MLNFL_ROOT_DIR, os.path.sep])
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
testYear = [2016]
trainYears = range(testYear[0]-3,testYear[0])

# training data set - includes one extra year for prev yr record
seasons = np.array(trainYears)
logging.info("training seasons >> {0}".format(seasons))

# get training data
# 1 - read all the games
path_to_lines = dataRoot + "lines/"
df_all_historical_games = madden.readGamesAll(path_to_lines, seasons)
# 2 - compile season record for all teams
df_all_teams = madden.seasonRecord(df_all_historical_games, reference_data)
# 3 - apply season records and compute other fields for all games
df_all_historical_games = madden.processGames(df_all_historical_games, df_all_teams, reference_data)
# 4 - remove extra year of data
df_all_historical_games = df_all_historical_games[df_all_historical_games.season.isin(seasons)]

# use different test set
season_test = np.array(testYear) # should be only one year
print(season_test)
logging.info("results for >> {0}".format(season_test))
# 1 - read all the games
dfGamesTest = madden.readGamesAll(path_to_lines, season_test)
# 2 - compile season record for all teams
dfTeamsTest = madden.seasonRecord(dfGamesTest, reference_data)
# 3 - apply season records and compute other fields for all games
dfGamesTest = madden.processGames(dfGamesTest, dfTeamsTest, reference_data)
# 4 - remove extra year of data
dfGamesTest = dfGamesTest[dfGamesTest.season.isin(season_test)]

# run the classifier
random_state = 11
svm_classifier = svm.SVC(kernel='poly', probability=True, random_state=random_state)
lr_classifier = linear_model.LogisticRegression(C=1e5)

svm_trained_classifier = madden.runScikitClassifier(df_all_historical_games, madden.FEATURE_COLUMNS, svm_classifier)
lr_trained_classifier = madden.runScikitClassifier(df_all_historical_games, madden.FEATURE_COLUMNS, lr_classifier)

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
df_svm_predict = madden.predictGames(dfTest, svm_trained_classifier, madden.FEATURE_COLUMNS)
# apply ranking logic and determine scoring outcomes for league
df_all_picks = madden.rankGames(df_svm_predict, reference_data, season_test[0])

# display weekly ranking output

# ranking methods choices
# 0. pick based on spread
# 1. always pick favored team, rank by probability of win
# 2. pick winner based on abs(probability - .5), rank by probability
# 3. pick winner based on abs(probability - .5), rank by abs(probability - .5)

DISPLAY_COLUMNS = ['season','gameWeek','Visitor','visitorRecord','Home Team','homeRecord',
            'Line','prevFavoredRecord','prevUnderdogRecord','predict_proba',
            'lineGuess','probaGuess', 'probaAbsGuess', 'predictTeam']


df_all_picks['predictTeam'] = np.where((df_all_picks['predict_proba'] - .5) > 0 , df_all_picks['favorite'], df_all_picks['underdog'])
guessCol = 'probaGuess'
predictCols = ['gameWeek','predictTeam', 'predict_proba', guessCol, 'favorite','lineGuess', 'Line']

print("\nPicks for week {0:0>2} using SVM\n".format(week_number))
svm_picks_df = df_all_picks[predictCols].sort_values(guessCol, ascending=False).copy()
print(svm_picks_df)

svm_out_file = "".join([args.picks_dir, os.path.sep, "svm_picks_week_{0:0>2}.csv".format(week_number)])
logging.info("Writing SVM output to {}...".format(svm_out_file))
svm_picks_df.to_csv(svm_out_file, index=False)

#week_filter = df_all_picks.gameWeek == week_number
#print("\nPicks using SVM")
#print(df_all_picks[week_filter][predictCols].sort(guessCol, ascending=False))
###################################################################################################################

###################################################################################################################
# apply results of logistic regression to the test set
df_lr_predict = madden.predictGames(dfTest, lr_trained_classifier, madden.FEATURE_COLUMNS)
# apply ranking logic and determine scoring outcomes for league
df_all_picks = madden.rankGames(df_lr_predict, reference_data, season_test[0])

# Use Method 2
df_all_picks['predictTeam'] = np.where((df_all_picks['predict_proba'] - .5) > 0 , df_all_picks['favorite'], df_all_picks['underdog'])

print("\nPicks for week {0:0>2} using LogReg\n".format(week_number))
log_reg_picks_df = df_all_picks[predictCols].sort_values(guessCol, ascending=False).copy()
print(log_reg_picks_df)

log_reg_out_file = "".join([args.picks_dir, os.path.sep, "log_reg_picks_week_{0:0>2}.csv".format(week_number)])
logging.info("Writing logistic regression output to {}...".format(log_reg_out_file))
log_reg_picks_df.to_csv(log_reg_out_file, index=False)

#week_filter = df_all_picks.gameWeek == week_number
#print("\nPicks using LogReg\n")
#print(df_all_picks[week_filter][predictCols].sort(guessCol, ascending=False))
###################################################################################################################

# display weekly ranking output for spread method

# ranking methods choices
# 0. pick based on spread
# 1. always pick favored team, rank by probability of win
# 2. pick winner based on abs(probability - .5), rank by probability
# 3. pick winner based on abs(probability - .5), rank by abs(probability - .5)

predictCols = ['favorite','lineGuess', 'absLine', 'Line', 'favoredHomeGame', 'divisionGame', 'favoredRecord']

sortCols = ['absLine','favoredHomeGame', 'divisionGame', 'favoredRecord', 'favorite']
df_spread = df_all_picks[predictCols].sort_values(sortCols , ascending=False)

print("\nPicks for week {0:0>2} using Spread\n".format(week_number))
print(df_spread)
spread_out_file = "".join([args.picks_dir, os.path.sep, "spread_picks_week_{0:0>2}.csv".format(week_number)])
logging.info("Writing spread output to {}...".format(spread_out_file))
df_spread.to_csv(spread_out_file, index=False)

