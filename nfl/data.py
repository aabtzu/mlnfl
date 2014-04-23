import pandas
import dateutil.parser as dp
import datetime
import numpy as np 
from sklearn import linear_model

import lookup

MAX_WEEK = 17

def getWeek(seasonStart,gameDateStr):
# determine week of season
	gameDate = dp.parse(gameDateStr).date()
	week = int(np.ceil((gameDate - seasonStart).days / 7.0))+1
	return week

def sameDivision(team1,team2,olookups):
# determine if two teams are in the same division

    conf1 = olookups.lookupCSV('teams',team1,'league')
    conf2 = olookups.lookupCSV('teams',team2,'league')
    
    div1 = olookups.lookupCSV('teams',team1,'division')
    div2 = olookups.lookupCSV('teams',team2,'division')
    
    if conf1 == conf2 and div1 == div2:
        return True
    else:
        return False
    
def readGames(dataRoot,season):
	# read a csv file of game scores and spreads into a pandas data frame
	#

	dataFile = dataRoot + "lines/nfl%slines.csv" % season
	dfAllGames = pandas.read_csv(dataFile)
	return dfAllGames

def seasonRecord(dfAllGames,olookups):
	# compile season stats by team
	# games played, games won, lost, record to date, home/away, division game, 
	#

	# list of teams to loop over
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

	    # determine if division game
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
	# lookup start of season for a given year
	seasonStr = olookups.lookupCSV('seasons',season,'start')
	seasonStart = dp.parse(seasonStr).date()
	return seasonStart

def getRecord(dfSeason,team,week):
	# get record of team by week in season
	# n.b. the week in season is NOT the same as the number of games played because of bye weeks
	record = float(dfSeason[(dfSeason['team'] == team) & (dfSeason['gamesPlayed'] == week-1)]['winPct'])
	return record

def processGames(dfAllGames, dfAllTeams, olookups, season, dfPrevSeason=None):
	# apply season record and other stats to all games played
	#

	seasonStart = getSeasonStart(olookups,season)

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

	# loop over each game and apply season record to date
	for gg in dfAllGames.iterrows():
	    game = gg[1] # need this because of how the generator works ??
	    gameDateStr = game['Date']
	    gameWeek = getWeek(seasonStart,gameDateStr)
	    
	    # get record from previous week 
	    if gameWeek > 1:
	        homeRecord = getRecord(dfAllTeams,game['Home Team'],gameWeek)
	        visitorRecord = getRecord(dfAllTeams,game['Visitor'],gameWeek)
	        
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
	    
	    # score, win and line info for each game
	    homeWin = int(int(game['Home Score']) > int(game['Visitor Score']))
	    scoreDiff = int(game['Home Score'] - game['Visitor Score'])
	    favoredWin = int((game['Line'] * scoreDiff) > 0 )
	    divGame = int(sameDivision(game['Home Team'].lower(),game['Visitor'].lower(),olookups))
	    favoredHomeGame = int(game['Line'] > 0)
	    
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
	    if dfPrevSeason is not None:
	    	prevHomeRecord = getRecord(dfPrevSeason,game['Home Team'],MAX_WEEK)
	       	prevVisitorRecord = getRecord(dfPrevSeason,game['Visitor'],MAX_WEEK)
	       	if game['Line'] > 0:
	       		prevFavoredRecord = prevHomeRecord
	       		prevUnderdogRecord = prevVisitorRecord
	       	else:
	       		prevFavoredRecord = prevVisitorRecord
	       		prevUnderdogRecord = prevHomeRecord
	    	
	    	prevFavRecord.append(prevFavoredRecord)
	    	prevDogRecord.append(prevUnderdogRecord)
	       	
	    
	    # print gameDateStr, gameWeek, game['Home Team'], game['Visitor'], homeRecord, visitorRecord
	
	# fill in data frame with all new columns    
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
	
	# need abs to rank easily
	dfAllGames['absLine'] = abs(dfAllGames['Line'])

	return dfAllGames

def getTrainData(dfAllGames,featuresList,maxTrainWeek=17):
	# extract features and classifiers from data frame for sklearn routines
	#
	# set maxTrainWeek if you dont want to use all weeks of the season
	# last week of season is often meaningless 

	dfTrain = dfAllGames[dfAllGames.gameWeek <= maxTrainWeek]
	y = dfTrain.favoredWin.tolist()
	X = dfTrain[featuresList].as_matrix()

	return X,y

def runLogisticRegression(X,y):
	# get logistic regression object for features/classifiers

	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(X,y)	

	# compute training accuracy
	sc = logreg.score(X,y)
	print "training data accuracy = ", sc

	return logreg

def runML(dfAllGames,featuresList,maxTrainWeek=17):
	# run machine learning on training data

	train_X,train_y = getTrainData(dfAllGames,featuresList,maxTrainWeek)
	logreg = runLogisticRegression(train_X,train_y)
	return logreg

def predictGames(dfAllGames,logreg, featuresList):
	# apply results of logistic regression to test data
	#
	dfPredict = dfAllGames
	predict_X, predict_y = getTrainData(dfAllGames,featuresList)

	# get results from pre-computes logreg object
	p = logreg.predict(predict_X) # 0/1 classifier
	pp = logreg.predict_proba(predict_X) # probablity of favored team winning
	dfxn = logreg.decision_function(predict_X) # not sure how this is helpful yet

	dfPredict['predict_proba'] = pp[:,1]
	dfPredict['predict_proba_abs'] = abs(pp[:,1] - 0.5) # 
	dfPredict['decision_fxn'] = dfxn

	scoreDiff = dfPredict['Home Score'] - dfPredict['Visitor Score']
	# predicted win = if the team that is predicted to win by logreg actually wins
	#                 not necessarily the favored team
	dfPredict['predictWin'] = 1 * ((scoreDiff * dfPredict['Line'] * (dfPredict['predict_proba'] - 0.5) )> 0)
	return dfPredict

def rankGames(dfPredict,olookups,season):
	# deterime weekly rankings for games based on predict from logistic regression
	#

	winningScore = int(olookups.lookupCSV('seasons',season,'winner'))
	weeks = dfPredict.gameWeek.unique()
	dfAll = None

	for ww in weeks:
		iw = dfPredict[dfPredict.gameWeek == ww].index
		dfWeek = dfPredict.loc[iw]
		nw = len(iw)
		dfLine = dfWeek.sort(['absLine','favoredHomeGame','predict_proba'])
		# print ww,nw
	    
	    # determine guess 
	    # several possibilities for predicting by probability
	    # 1. always pick favored team, rank by probablity of win
	    # 2. pick winner based on abs(probability - .5), rank by probability
	    # 3. pick winner based on abs(probability - .5), rank by abs(probability - .5)
		dfLine['lineGuess'] = dfLine['absLine'].rank('first') + (16 - nw)
		dfLine['probaGuess'] = dfLine['predict_proba'].rank('first') + (16 - nw)
		dfLine['probaAbsGuess'] = dfLine['predict_proba_abs'].rank('first') + (16 - nw)
	    
	    # determine score
		dfLine['lineScore'] = dfLine['lineGuess'] * dfLine['favoredWin']

		dfLine['probaScore1'] = dfLine['probaGuess'] * dfLine['favoredWin']  
		dfLine['probaScore2'] = dfLine['probaGuess'] * dfLine['predictWin']  
		dfLine['probaScore3'] = dfLine['probaAbsGuess'] * dfLine['predictWin']  

		if ww == 1:
			dfAll = dfLine
		else:
			dfAll = dfAll.append(dfLine)	


	dispCols = ['gameWeek','Visitor','Home Team','Line','predict_proba','Visitor Score','Home Score','favoredWin','predictWin','lineGuess','probaGuess', 'lineScore','probaScore']
	#print dfAll[dfAll.gameWeek==1][dispCols]

	g = dfAll.groupby('gameWeek')['lineScore','probaScore1','probaScore2','probaScore3'].sum()
	print g
	dd = [g.sum(), g.sum() - winningScore]
	ss = pandas.DataFrame(dd).transpose()
	ss[2]  = ss[1] > 0 
	ss.columns = ['score','win by','win']
	print ss



	return dfAll


