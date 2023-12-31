from __future__ import division
from __future__ import print_function

__author__ = 'alainledon'

WEEK_TO_PICK = 1

import os
import numpy as np
import pandas as pd

import logging
import argparse

from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble

from nfl.referencedata import ReferenceData
from nfl import madden
from nfl import probability

pd.options.mode.chained_assignment = None
pd.set_option('expand_frame_repr', False)

# get the working directory from the environment variable MLNFL_ROOT
MLNFL_ROOT_DIR = os.environ['MLNFL_ROOT']
print(MLNFL_ROOT_DIR)

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger()


def get_picks_by_spread():
    pass


def get_picks_by_svm():
    pass


def get_picks_by_log_reg():
    pass


def main(season, week_number, picks_dir):
    # define the root directory for the nfl code in $MLNLF_ROOT
    code_dir = "".join([MLNFL_ROOT_DIR, os.path.sep])
    data_root = "".join([code_dir, "data", os.path.sep])

    os.chdir(code_dir)

    logging.info("Base directory = {0}".format(code_dir))
    logging.info("Data directory = {0}".format(data_root))

    # location of lookup files

    lookupFiles = {'teams' : {'file': 'nflTeams.csv'}, 'seasons': {'file': 'seasons.csv'}, }

    lookupDir = "".join([data_root, 'lookup', os.path.sep])

    logging.info(f"lookupFiles = {lookupFiles}")
    logging.info(f"lookupDir = {lookupDir}")

    # load reference data
    reference_data = ReferenceData(lookupDir)

    # train on previous 3 yrs of data
    test_year = [season]
    train_years = list(range(test_year[0]-3, test_year[0]))

    # training data set - includes one extra year for prev yr record
    seasons = np.array(train_years)
    logging.info("training seasons >> {0}".format(seasons))

    # get training data
    # 1 - read all the games
    path_to_lines = data_root + "lines/"
    df_all_historical_games = madden.readGamesAll(path_to_lines, seasons)
    # 2 - compile season record for all teams
    df_all_teams = madden.seasonRecord(df_all_historical_games, reference_data)
    # 3 - apply season records and compute other fields for all games
    df_all_historical_games = madden.processGames(df_all_historical_games, df_all_teams, reference_data)
    # 4 - remove extra year of data
    df_all_historical_games = df_all_historical_games[df_all_historical_games.season.isin(seasons)]

    # use different test set
    season_test = np.array(test_year) # should be only one year
    print(season_test)
    logging.info("results for >> {0}".format(season_test))
    # 1 - read all the games
    df_games_test = madden.readGamesAll(path_to_lines, season_test)
    # 2 - compile season record for all teams
    df_teams_test = madden.seasonRecord(df_games_test, reference_data)
    # 3 - apply season records and compute other fields for all games
    df_games_test = madden.processGames(df_games_test, df_teams_test, reference_data)
    # 4 - remove extra year of data
    df_games_test = df_games_test[df_games_test.season.isin(season_test)]

    # get games for testing and predicting -- should be only one year
    season_test = np.array(test_year)

    # add current season games to training set for 2nd logreg model
    df_all_historical_games2 = df_all_historical_games.append(df_games_test[df_games_test.gameWeek < week_number])

    # pick only this week's games for predict
    dfTest = df_games_test[df_games_test.gameWeek == week_number]

    print(len(df_all_historical_games), len(df_all_historical_games2), len(dfTest))

    # run the classifier
    random_state = 11

    svm_classifier = svm.SVC(kernel='poly', probability=True, random_state=random_state)

    lr_classifier = linear_model.LogisticRegression(C=1e5)
    lr2_classifier = linear_model.LogisticRegression(C=1e5)

    lr_trained_classifier = madden.runScikitClassifier(df_all_historical_games, madden.FEATURE_COLUMNS, lr_classifier)
    #svm_trained_classifier = madden.runScikitClassifier(df_all_historical_games, madden.FEATURE_COLUMNS, svm_classifier)
    lr2_trained_classifier = madden.runScikitClassifier(df_all_historical_games2, madden.FEATURE_COLUMNS, lr2_classifier)

    ###################################################################################################################
    # apply results of logistic regression to the test set
    if 0:
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

        svm_out_file = "".join([picks_dir, os.path.sep, "svm_picks_week_{0:0>2}.csv".format(week_number)])
        logging.info("Writing SVM output to {}...".format(svm_out_file))
        svm_picks_df.to_csv(svm_out_file, index=False)

        #week_filter = df_all_picks.gameWeek == week_number
        #print("\nPicks using SVM")
        #print(df_all_picks[week_filter][predictCols].sort(guessCol, ascending=False))
    ###################################################################################################################

    ###################################################################################################################
    # apply results of logistic regression to the test set
    print(dfTest)
    df_lr_predict = madden.predictGames(dfTest, lr_trained_classifier, madden.FEATURE_COLUMNS)
    # apply ranking logic and determine scoring outcomes for league
    df_all_picks = madden.rankGames(df_lr_predict, reference_data, season_test[0])

    # Use Method 2
    df_all_picks['predictTeam'] = np.where((df_all_picks['predict_proba'] - .5) > 0 , df_all_picks['favorite'], df_all_picks['underdog'])
    guessCol = 'probaGuess'
    predictCols = ['gameWeek', 'predictTeam', 'predict_proba', guessCol, 'favorite', 'lineGuess', 'Line']

    print("\nPicks for week {0:0>2} using LogReg\n".format(week_number))
    log_reg_picks_df = df_all_picks[predictCols].sort_values(guessCol, ascending=False).copy()
    print(log_reg_picks_df)

    log_reg_out_file = "".join([picks_dir, os.path.sep, "log_reg_picks_week_{0:0>2}.csv".format(week_number)])
    logging.info("Writing logistic regression output to {}...".format(log_reg_out_file))
    log_reg_picks_df.to_csv(log_reg_out_file, index=False)



    if 0:
        ###################################################################################################################
        # legreg version 2
        # apply results of logistic regression to the test set
        df_lr2_predict = madden.predictGames(dfTest, lr2_trained_classifier, madden.FEATURE_COLUMNS)
        # apply ranking logic and determine scoring outcomes for league
        df_all_picks = madden.rankGames(df_lr2_predict, reference_data, season_test[0])

        # Use Method 2
        df_all_picks['predictTeam'] = np.where((df_all_picks['predict_proba'] - .5) > 0 , df_all_picks['favorite'], df_all_picks['underdog'])

        print("\nPicks for week {0:0>2} using LogReg2\n".format(week_number))
        log_reg_picks_df = df_all_picks[predictCols].sort_values(guessCol, ascending=False).copy()
        print(log_reg_picks_df)

        log_reg_out_file = "".join([picks_dir, os.path.sep, "log_reg_2_picks_week_{0:0>2}.csv".format(week_number)])
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

    # add probability

    df_lines_performance = probability.load_probability()
    df_spread = df_spread.merge(df_lines_performance, left_on="Line", right_on='line', how='left')


    print("\nPicks for week {0:0>2} using Spread\n".format(week_number))
    print(df_spread)
    spread_out_file = "".join([picks_dir, os.path.sep, "spread_picks_week_{0:0>2}.csv".format(week_number)])
    logging.info("Writing spread output to {}...".format(spread_out_file))
    df_spread.to_csv(spread_out_file, index=False)

    return df_spread


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--week', '-w', action='store', default=WEEK_TO_PICK, dest='game_week',
                        type=int, help='Pass the week number to make the picks')
    parser.add_argument('--directory', '-d', action='store', default="".join([MLNFL_ROOT_DIR, os.path.sep, 'picks']),
                        dest='picks_dir', help='Pass the target directory to get a csv with the picks')
    args = parser.parse_args()

    picks_dir = args.picks_dir

    # predict one week of current season
    week_number = args.game_week
    season = 2020
    main(season, week_number, picks_dir)

