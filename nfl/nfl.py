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
import referencedata

lookupFiles = {'teams': {'file': 'nflTeams.csv'},
               'seasons': {'file': 'seasons.csv'},
}

lookupDir = dataRoot + 'lookup/'
olookups = referencedata.ReferenceData(lookupDir, lookupFiles)

#olookups.seasons_teams['seasons'].keys()

# <codecell>

import madden

reload(madden)

season = '2013'
# read data file
dfAllGames = madden.readGames(dataRoot, season)
# compile season record for all teams
dfAllTeams = madden.seasonRecord(dfAllGames, olookups)
# apply season records and compute other fields for all games
dfAllGames = madden.processGames(dfAllGames, dfAllTeams, olookups, season)
# run the logistic regression
logreg = madden.runML(dfAllGames)

# use different test set 
season = '2013'
dfGamesTest = madden.readGames(dataRoot, season)
dfTeamsTest = madden.seasonRecord(dfGamesTest, olookups)
dfGamesTest = madden.processGames(dfGamesTest, dfTeamsTest, olookups, season)

# apply results of logistic regression to the test set
dfPredict = madden.predictGames(dfGamesTest, logreg)
dfAll = madden.rankGames(dfPredict)

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


