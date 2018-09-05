import csv
import pandas as pd
import dateutil.parser as dp

SEASONS_FILE = "seasons.csv"
TEAMS_FILE = "nflteams.csv"


def loadSimpleCSV(source, dataDir, lookupDict, lookupCol=None):
    """
    :Synopsis: loads multiple files

    :param source: team1
    :param dataDir: team1
    :param lookupDict: Lookup object with a "teams" dictionary
    :param lookupCol: team1

    :returns: True if the teams are in the same division
    """
    f = dataDir + lookupDict[source]['file']
    reader = csv.DictReader(open(f))
    if lookupCol is None:
        fields = reader.fieldnames
        lookupCol = fields[0]

    dmaDict = dict()
    for r in reader:
        if isinstance(lookupCol, tuple):
            k = tuple()
            for ff in lookupCol:
                k = k + (r[ff].lower(),)
        else:
            k = r[lookupCol].lower()
        dmaDict[k] = r

    return dmaDict


class ReferenceData(object):
    """
    :Synopsis: Class to do lookups

    Responsibility -- To do a "vlookup" on a csv file
    """

    def __init__(self, dataDir, seasonsFile = SEASONS_FILE, teamsFile = TEAMS_FILE):
        """
        :Synopsis: Loads the seasons and teams files from

        :param dataDir: Directory where the files are located

        :returns: seasons and teams data frames loaded
        """
        # Read all seasons and teams from the data files
        self.seasons_df = pd.read_csv("".join([dataDir, seasonsFile]), index_col='season')
        self.teams_df = pd.read_csv("".join([dataDir, teamsFile]), index_col='team')

    def get_team_league(self, team):
        """
        :Synopsis: Returns the team's league

        :param team: The name of the team as a string

        :returns: The team's league
        """
        return self.teams_df.loc[team, 'league']

    def get_team_division(self, team):
        """
        :Synopsis: Returns the team's division

        :param team: The name of the team as a string

        :returns: The team's division
        """
        return self.teams_df.loc[team, 'division']

    def getSeasonStartDate(self, season):
        """
        returns the date of the first game of the season as a date
        """
        return dp.parse(self.seasons_df.loc[int(season), "start"]).date()

    def getSeasonWinner(self, season):
        """
        Returns the ID of the winner on a given season.
        """
        return int(self.seasons_df.loc[int(season), "winner"])