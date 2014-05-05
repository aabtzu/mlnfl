import csv


def loadSimpleCSV(source, dataDir, lookupDict, lookupCol=None):
    """
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


class Lookup:
    """
    :Synopsis: Class to do lookups

    Responsibility -- To do a "vlookup" on a csv file
    """

    def __init__(self, dataDir, lookupDict):
        # init the lookup files
        self.ldict = dict()

        # loop over all the lookup files
        for k in lookupDict.keys():
            try:
                lookupCol = lookupDict[k]['lookupCol']
            except:
                lookupCol = None
            self.ldict[k] = loadSimpleCSV(k, dataDir, lookupDict, lookupCol)


    def lookupCSV(self, lsource, lval, lcol):
        # find a value by source, key, lookup column
        try:
            val = self.ldict[lsource][lval][lcol]
        except:
            val = None
        return val

    def getLookupTable(self, source):
        # return the entire lookup dict for a specific source
        return self.ldict[source]

    def getKeys(self, source, conditionList=None):
        # return all the keys of the lookup dict for a specific source
        return self.ldict[source].keys()

    def getKeysSorted(self, source, lookupKey, sortField, sortNum=False):
        # return a sorted list of all the keys of the lookup dict for a specific source

        keyList = self.getList(source, sortField, sortNum)
        keyOrder = [k[lookupKey] for k in keyList]

        return keyOrder

    def getList(self, source, sortField=None, sortNum=False):
        # return a sorted list of all the values of the lookup dict for a specific source
        llist = list()
        for k, v in self.ldict[source].iteritems():
            llist.append(v)

        if sortField is not None:
            if sortNum:
                sortList = sorted(llist, key=lambda f: int(f[sortField]))
            else:
                sortList = sorted(llist, key=lambda f: f[sortField].lower())

            llist = sortList

        return llist