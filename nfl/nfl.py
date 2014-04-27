# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

homeDir = os.environ['HOME'] + '/'
codeDir = homeDir + 'repos/mlkaggle/nfl/'
dataRoot = codeDir + "data/"
os.chdir(codeDir)

dataRoot

# <codecell>

# lookup files
import lookup

lookupFiles = {'teams': {'file': 'nflTeams.csv'},
               'seasons': {'file': 'seasons.csv'},
}

lookupDir = dataRoot + 'lookup/'
olookups = lookup.Lookup(lookupDir, lookupFiles)

#olookups.ldict['seasons'].keys()

# <codecell>

import data

reload(data)

season = '2013'
# read data file
dfAllGames = data.readGames(dataRoot, season)
# compile season record for all teams
dfAllTeams = data.seasonRecord(dfAllGames, olookups)
# apply season records and compute other fields for all games
dfAllGames = data.processGames(dfAllGames, dfAllTeams, olookups, season)
# run the logistic regression
logreg = data.runML(dfAllGames)

# use different test set 
season = '2013'
dfGamesTest = data.readGames(dataRoot, season)
dfTeamsTest = data.seasonRecord(dfGamesTest, olookups)
dfGamesTest = data.processGames(dfGamesTest, dfTeamsTest, olookups, season)

# apply results of logistic regression to the test set
dfPredict = data.predictGames(dfGamesTest, logreg)
dfAll = data.rankGames(dfPredict)

# display details for a single week
dispWeek = 1
dispCols = ['gameWeek', 'Visitor', 'Home Team', 'Line', 'predict_proba', 'Visitor Score', 'Home Score', 'favoredWin',
            'lineGuess', 'probaGuess', 'lineScore', 'probaScore']
dfAll[dfAll.gameWeek == dispWeek][dispCols]

# <codecell>

# diagnotics ... display all columns
dfAllGames.columns

# <codecell>

# diagnostics ... plot outcomes of games
nWin = dfAllGames.favoredWin.index.tolist()
nLose = dfAllGames[dfAllGames['favoredWin'] == 0].index.tolist()

plot(dfAllGames.favoredRecord[nWin], dfAllGames.underdogRecord[nWin], 'bo')
plot(dfAllGames.favoredRecord[nLose], dfAllGames.underdogRecord[nLose], 'ro')

# <codecell>

# diagnostics ... print out end of season record
cols = ['team', 'gamesWon', 'gamesLost']
dfAllTeams[dfAllTeams.gamesPlayed == 16][cols].sort('gamesWon', ascending=False)

# <codecell>


