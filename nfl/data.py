import pandas
import dateutil.parser as dp
import datetime
import numpy as np 
from sklearn import linear_model

import lookup


def getWeek(seasonStart,gameDateStr):
    gameDate = dp.parse(gameDateStr).date()
    week = int(np.ceil((gameDate - seasonStart).days / 7.0))+1
    return week

def sameDivision(team1,team2,olookups):
    conf1 = olookups.lookupCSV('teams',team1,'league')
    conf2 = olookups.lookupCSV('teams',team2,'league')
    
    div1 = olookups.lookupCSV('teams',team1,'division')
    div2 = olookups.lookupCSV('teams',team2,'division')
    
    #print team1,conf1,div1
    #print team2,conf2,div2
    if conf1 == conf2 and div1 == div2:
        return True
    else:
        return False
    

def readGames(dataRoot,season):

	dataFile = dataRoot + "lines/nfl%slines.csv" % season
	dfAllGames = pandas.read_csv(dataFile)
	return dfAllGames



def seasonRecord(dfAllGames,olookups):

# compile season stats by team
# games played, games won, lost, record to date, home/away, division game, 

	teams = dfAllGames.Visitor.unique()

	dfAllTeams = None
	for ii,tt in enumerate(teams):
	    # print ii,tt
	    dfTeam = dfAllGames[(dfAllGames.Visitor == tt) | (dfAllGames['Home Team'] == tt)]
	    dfTeam['gamesPlayed'] = range(1,len(dfTeam.index)+1) 
	    dfTeam['team'] = tt
	    dfTeam['homeGame'] = dfAllGames['Home Team'] == tt
	    dfTeam['wonGame'] = ((dfTeam['Visitor Score'] < dfTeam['Home Score']) & dfTeam['homeGame']) | ((dfTeam['Visitor Score'] > dfTeam['Home Score']) & (dfTeam['homeGame'] == False))
	    dfTeam['gamesWon'] = dfTeam['wonGame'].cumsum() 
	    dfTeam['homeGamesWon'] = (dfTeam['wonGame'] & dfTeam['homeGame']).cumsum()
	    dfTeam['gamesLost'] = dfTeam['gamesPlayed'] - dfTeam['gamesWon']
	    dfTeam['winPct'] = dfTeam['gamesWon'] / dfTeam['gamesPlayed']
	    dfTeam['homeGamesPlayed'] = dfTeam['homeGame'].cumsum()
	    dfTeam['homeWinPct'] = dfTeam['homeGamesWon'] / dfTeam['homeGamesPlayed']

	    opponent = list()
	    divGame = list()
	    for gg in dfTeam.iterrows():
	        row = gg[1]
	        if row['Home Team'] == row['team']:
	            team2 = row['Visitor']
	        else:
	            team2 = row['Home Team']
	        
	        opponent.append(team2)
	        divGame.append(sameDivision(row['team'].lower(),team2.lower(),olookups))
	    
	    dfTeam['opponent'] = opponent
	    dfTeam['divGame'] = divGame    
	    
	    dfAllTeams = pandas.concat([dfAllTeams,dfTeam])
	return dfAllTeams

def getSeasonStart(olookups,season):

	seasonStr = olookups.lookupCSV('seasons',season,'start')
	seasonStart = dp.parse(seasonStr).date()

	return seasonStart

def processGames(dfAllGames, dfAllTeams, olookups, season):

	seasonStart = getSeasonStart(olookups,season)

	favoredHome = list()
	homePct = list()
	visitorPct = list()
	gameNum = list()
	winner = list()
	favored = list()
	division = list()
	favRecord = list()
	dogRecord = list()

	for gg in dfAllGames.iterrows():
	    #print gg[1]
	    game = gg[1]
	    gameDateStr = game['Date']
	    gameWeek = getWeek(seasonStart,gameDateStr)
	    
	    # get record from previous week 
	    if gameWeek > 1:
	        homeRecord = float(dfAllTeams[(dfAllTeams['team'] == game['Home Team']) & (dfAllTeams['gamesPlayed'] == gameWeek-1)]['winPct'])
	        visitorRecord = float(dfAllTeams[(dfAllTeams['team'] == game['Visitor']) & (dfAllTeams['gamesPlayed'] == gameWeek-1)]['winPct'])
	        
	        if game['Line'] > 0:
	            favoredRecord = homeRecord
	            underdogRecord = visitorRecord
	        else:
	            favoredRecord = visitorRecord
	            underdogRecord = homeRecord
	    else:
	        homeRecord = 0.0
	        visitorRecord = 0.0
	        favoredRecord = 0.0
	        underdogRecord = 0.0
	    
	    
	    homeWin = int(int(game['Home Score']) > int(game['Visitor Score']))
	    scoreDiff = int(game['Home Score'] - game['Visitor Score'])
	    favoredWin = int((game['Line'] * scoreDiff) > 0 )
	    divGame = int(sameDivision(game['Home Team'].lower(),game['Visitor'].lower(),olookups))
	    favoredHomeGame = int(game['Line'] > 0)
	    
	    #print int(game['Home Score']), int(game['Visitor Score']), homeWin, scoreDiff, game['Line']
	    favoredHome.append(favoredHomeGame)
	    homePct.append(homeRecord)
	    visitorPct.append(visitorRecord)
	    gameNum.append(gameWeek)    
	    winner.append(homeWin)
	    favored.append(favoredWin)
	    division.append(divGame)
	    favRecord.append(favoredRecord)
	    dogRecord.append(underdogRecord)
	    
	    # get record from previous season
	    
	    
	    # print gameDateStr, gameWeek, game['Home Team'], game['Visitor'], homeRecord, visitorRecord
	    
	dfAllGames['favoredHomeGame'] = favoredHome    
	dfAllGames['divisionGame'] = division
	dfAllGames['homeWin'] = winner
	dfAllGames['favoredWin'] = favored
	dfAllGames['gameWeek'] = gameNum
	dfAllGames['homeRecord'] = homePct    
	dfAllGames['visitorRecord'] = visitorPct

	dfAllGames['favoredRecord'] = favRecord
	dfAllGames['underdogRecord'] = dogRecord
	dfAllGames['absLine'] = abs(dfAllGames['Line'])


	return dfAllGames

def getTrainData(dfAllGames,maxTrainWeek=17):

	dfTrain = dfAllGames[dfAllGames.gameWeek <= maxTrainWeek]
	y = dfTrain.favoredWin.tolist()

	# define indep variables for logistic regression
	xcols = ['favoredRecord','underdogRecord','gameWeek','absLine','divisionGame','favoredHomeGame']
	X = dfTrain[xcols].as_matrix()

	return X,y

def runLogisticRegression(X,y):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(X,y)	

	sc = logreg.score(X,y)
	print "training data accuracy = ", sc

	return logreg

def runML(dfAllGames,maxTrainWeek=17):

	train_X,train_y = getTrainData(dfAllGames,maxTrainWeek)
	logreg = runLogisticRegression(train_X,train_y)
	return logreg

def predictGames(dfAllGames,logreg):

	dfPredict = dfAllGames
	predict_X, predict_y = getTrainData(dfAllGames)

	p = logreg.predict(predict_X)
	pp = logreg.predict_proba(predict_X)
	dfxn = logreg.decision_function(predict_X)

	dfPredict['predict'] = p
	dfPredict['predict_proba'] = pp[:,1]
	dfPredict['decision_fxn'] = dfxn

	return dfPredict

def rankGames(dfPredict):

	weeks = dfPredict.gameWeek.unique()

	dfAll = None

	for ww in weeks:
	    iw = dfPredict[dfPredict.gameWeek == ww].index
	    dfWeek = dfPredict.loc[iw]
	    nw = len(iw)
	    dfLine = dfWeek.sort(['absLine','favoredHomeGame','predict_proba'])
		# print ww,nw
	    
	    # determine guess 
	    dfLine['lineGuess'] = dfLine['absLine'].rank('first') + (16 - nw)
	    dfLine['probaGuess'] = dfLine['predict_proba'].rank('first') + (16 - nw)
	    
	    # determine score
	    dfLine['lineScore'] = dfLine['lineGuess'] * dfLine['favoredWin']
	    dfLine['probaScore'] = dfLine['probaGuess'] * dfLine['favoredWin']
	    
	    if ww == 1:
	        dfAll = dfLine
	    else:
	        dfAll = dfAll.append(dfLine)	


	dispCols = ['gameWeek','Visitor','Home Team','Line','predict','predict_proba','Visitor Score','Home Score','favoredWin','lineGuess','probaGuess', 'lineScore','probaScore']
	#print dfAll[dfAll.gameWeek==1][dispCols]

	g = dfAll.groupby('gameWeek')['lineScore','probaScore'].sum()
	print g
	print g.sum()

	return dfAll


