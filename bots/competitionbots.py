from __future__ import division
from genericpath import exists
from token import EQUAL
from player import Bot
from game import *
from util import Variable
import random
import itertools
import logging
import collections
import math
import logging
import logging.handlers













class Jerry(Bot):

    def onGameRevealed(self, players, spies):
        self.leaderCountdown = 10
        self.spies = spies
        self.players = players
        self.configurations = {k: 0.0 for k in permutations([True, True, False, False])}

    def onMissionAttempt(self, mission, tries, leader):
        if self == leader:
            self.leaderCountdown = 4
        else:
            self.leaderCountdown -= 1

    def select(self, players, count):
        lowest = min(self.configurations.itervalues())
        bestOptions = [k for k, v in self.configurations.items() if v == lowest]
        choice = random.choice(bestOptions)
        teammates = [p for p, s in zip(self.others(), choice) if not s]
        return [self] + random.sample(teammates, count-1)

    def onTeamSelected(self, leader, team):
        for c in self.configurations:
            unlikely = 0.0
            spies = self.getSpies(c)
            spiesOnMission = self.getSpiesOnMission(c)
            if leader in spies and len(spiesOnMission) > 1:
                unlikely += 1.0
            if leader in spies and len(spiesOnMission) == 0:
                unlikely += 3.0
            if leader not in spies and len(spiesOnMission) == len(team):
                unlikely += 3.0
            if leader not in spies and len(spiesOnMission) + 1 == len(team):
                unlikely += 0.0
            self.configurations[c] += unlikely

    def expertVote(self, team):
        if self.spy:
            if self.game.tries == 5:
                return True
            spiesOnMission = [s for s in team if s in self.spies]
            if len(spiesOnMission) > 0 and (self.game.wins == 2 or self.game.losses == 2):
               return True
            if len(spiesOnMission) > 0:
                if self not in team and self.game.leader not in self.spies:
                    return True
                if self not in team and self.game.leader in self.spies and len(team) < 3:
                    return True
            if len(team) == 3:
                return self in team
        else:
            if self.game.leader == self:
                return True
            if self.game.tries == 5:
                return True
        return None

    def vote(self, team): 
        if self.expertVote(team) is not None:
            return self.expertVote(team)

        u = []
        for c in self.configurations:
            if len([s for s in team if s in self.getSpies(c)]) == 0:
                u.append(self.configurations[c])
        if len(u) == 0:
            if self in self.spies:
                return True if random.uniform(0,1) < 0.4 else False
            else:
                return False
        threshold = min(self.configurations.values())
        if self in self.spies:
            if len([s for s in team if s in self.spies]) == 1:
                return True if random.uniform(0,1) < 0.4 else False
        return min(u) <= threshold

    def onVoteComplete(self, votes):
        for c in self.configurations:
            unlikely = 0.0
            spies = self.getSpies(c)
            spiesOnMission = self.getSpiesOnMission(c)
            for p, v in zip(self.game.players, votes):
                if p in spies and len(spiesOnMission) == 1 and not v:
                    unlikely += 1.0
                if p in spies and len(spiesOnMission) == 0 and v:
                    unlikely += 1.0
                if p in spies and len(spiesOnMission) > 1 and v:
                    unlikely += 0.5
                if p not in spies and self.game.tries == 5 and not v:
                    unlikely += 5.0
                if p not in spies and p not in self.game.team and v:
                    if len(self.game.team) <= 2:
                        unlikely += 1.0
                    if len(self.game.team) >= 3:
                        unlikely += 3.0
            self.configurations[c] += unlikely

    def sabotage(self):
        if self.game.wins == 2 or self.game.losses == 2:
            return True
        if len(self.game.team) == 2:
            if self.game.turn == 1:
                return False
            if self.game.turn == 2:
                return True if random.uniform(0,1) < 0.2 else False
        if len([s for s in self.game.team if s in self.spies]) == 2:
            if self.game.leader != self:
                return False
            if len(self.game.team) == 2:
                return False
            else:
                return True if random.uniform(0,1) < 0.3 else False
        if len([s for s in self.game.team if s in self.spies]) == 3:
            return False
        return True

    def onMissionComplete(self, sabotaged):
        for c in self.configurations:
            unlikely = 0.0
            spies = self.getSpies(c)
            spiesOnMission = self.getSpiesOnMission(c)
            if len(spiesOnMission) < sabotaged:
                unlikely += 100000.0
            self.configurations[c] += unlikely

    def onMissionFailed(self, leader, team):
        for c in self.configurations:
            unlikely = 0.0
            spies = self.getSpies(c)
            spiesOnMission = [s for s in team if s in spies]
            if leader in spies and len(spiesOnMission) > 1:
                unlikely += 1.0
            if leader in spies and len(spiesOnMission) == 0:
                unlikely += 2.0
            self.configurations[c] += unlikely

    def getSpies(self, config):
        return set([player for player, spy in zip(self.others(), config) if spy])

    def getSpiesOnMission(self, config):
        spies = self.getSpies(config)
        return [s for s in spies if s in self.game.team]




class GameCombinations:
    def __init__(self, playerCnt, spyCnt):
        #self.combinations = []
        #cnt = math.factorial(playerCnt)/(math.factorial(spyCnt)*math.factorial(playerCnt - spyCnt))
        #indices = [i for i in range(spyCnt)]
        #for i in range(cnt):
        #   self.combinations.append([True if x in indices else False for x in range(playerCnt)])
        #
        #   for j in range(spyCnt-1, -1, -1):
        #       if indices[j] < (playerCnt - spyCnt + j):
        #           indices[j] += 1
        #           for k in range(j+1, spyCnt):
        #               indices[k] = indices[k-1]+1
        #           break
        # next line do the same. Should work for any number of players and spies
        self.combinations = map(lambda x: [True if y in x else False for y in range(playerCnt)], itertools.combinations(range(playerCnt),spyCnt))
    
    def addSabotage(self, team, sabotage):
        # remove those combinations which are not possible for given combination of team and number of sabotages
        reducedCombinations = []
        for c in self.combinations:
            if self._countOfSpies(team, c) >= sabotage:
                reducedCombinations.append(c)
        self.combinations = reducedCombinations
        
    def _countOfSpies(self, team, row):
        r = 0
        for p in team:
            if row[p.index] == True:
                r += 1
        return r
                
    def getProbabilities(self):
        # returns list of probabilities for player calculated from rest team combinations
        cnt = len(self.combinations)
        res = [0 for i in range(len(self.combinations[0]))] 
        for c in self.combinations:
            for j in range(len(c)):
                if c[j]:
                    res[j] += 1
        return map(lambda x: float(x)/cnt, res)
        
class SampledValue:
    # class which calculates mean value from given samples
    def __init__(self, initValue):
        self.sampleCnt = 1
        self.sampleSum = initValue
    
    def add(self, value):
        self.sampleSum += value
        self.sampleCnt += 1
        
    def get(self):
        return 0 if (self.sampleCnt == 0) else (float(self.sampleSum) / float(self.sampleCnt))  
        
    def __repr__(self):
        return "Sample cnt: %0.1f sample sum: %0.1f" % (self.sampleCnt, self.sampleSum)
        
class SampledValueNonLinear:
    # calculated mean value from samples but gives more weight to values closer to extremes
    # so for values 0.5 and 0.9 returns 0.85 instead of 0.7
    def __init__(self, initValue):
        self.sampleCnt = 1
        self.sampleSum = initValue
    
    def add(self, value):
        weight = math.fabs(0.5-value)*50.0
        self.sampleSum += value * weight
        self.sampleCnt += weight
        
    def get(self):
        return 0 if (self.sampleCnt == 0) else (float(self.sampleSum) / float(self.sampleCnt))  
        
    def __repr__(self):
        return "Sample cnt: %0.1f sample sum: %0.1f" % (self.sampleCnt, self.sampleSum)
    
class PlayerInfo:
    # this class holds suspicions from player actions 
    def __init__(self, player):
        self.suspicionFromActions = SampledValueNonLinear(0.5)
        self.suspicion = 0.5
        self.player = player
        self.selectedPlayers = []
    
    def onSuspiciousAction(self, spyProb):
        self.suspicionFromActions.add(float(spyProb))
        
    def onTrustedAction(self, resProb):
        self.suspicionFromActions.add(float(1) - float(resProb))
        
    def onTeamSelected(self, team):
        self.selectedPlayers.append(team)
        
    def isSpy(self):
        return self.suspicion > 0.99
    
    def updateSuspicion(self, spyRatio):
        if spyRatio < 0.1 or spyRatio > 0.9:
            self.suspicion = spyRatio
        else:
            self.suspicion = (spyRatio + self.suspicionFromActions.get()*2.0)/3.0
    
    def __repr__(self):
        return "<%i-%s %0.2f %0.2f>" % (self.player.index, self.player.name, self.suspicionFromActions.get(), self.suspicion)
        
        
class PlayerStatistics:
    def __init__(self):
        # probability of player sabotage mission when is on it
        self.sabotage = SampledValue(0.8)
        # probability that player votes against the team in first turn
        self.spyVotesFalseFirstTurn = SampledValue(0.5)
        self.resVotesFalseFirstTurn = SampledValue(0.5)
        # probability that player votes for team with spy in it
        self.spyVotesForSpy = SampledValue(0.5)
        self.resVotesForSpy = SampledValue(0.5)
        # when spy selects players for mission, how often he selects another spy to team
        self.spySelectsOtherSpy = SampledValue(0.5)
        # in game of 5 players, what is probability that player who is not in team votes for team
        self.spyVotesForFullTeam = SampledValue(0.5)
        self.resVotesForFullTeam = SampledValue(0.5)

class ScepticBot(Bot):
    """Don't trust anyone."""

    playerStats = {}

    def onGameRevealed(self, players, spies):
        self.playerInfos = [PlayerInfo(p) for p in players if p.index != self.index]
        self.spies = spies
        
        spyCntDict = {5:2, 6:2, 7:3, 8:3, 9:3, 10:4}
        self.playersCnt = len(players)
        self.spiesCnt = spyCntDict[self.playersCnt]
        
        self.gameCombinations = GameCombinations(self.playersCnt, self.spiesCnt)
        if not self.spy:
            others = [p for p in players if p.index != self.index]
            # remove all combinations containing me as a spy from combinations
            self.gameCombinations.addSabotage(others, self.spiesCnt)
        
        for p in players:
            self.playerStats.setdefault(p.name, PlayerStatistics())

        # members to store info about game flow
        self.missions = []
        self.votes = []
        self.selections = []
        
    def select(self, players, count):
        #if self.game.turn == 1:
        #   # in first turn, select other players to try force spies to reveal themselves
        #   others = [p.player for p in self.playerInfos if p.player != self]
        #   others.sort(key=lambda x: self._getSpyRatio(x, 'spyVotesFalseFirstTurn', 'resVotesFalseFirstTurn'))
        #   return others[:count]
        # this idea didn't work at all
        
        # playerInfos are sorted by suspicion value, select less suspicious players + me
        others = [p.player for p in self.playerInfos if p.player not in self.spies]
        me = [p for p in players if p.index == self.index]
        if self.spy:
            # spy selects most suspicious resistance players
            return me + others[len(others)-(count-1):]
        else:
            # resistance selects less suspicious players
            return me + others[:count -1]

    def onTeamSelected(self, leader, team):
        # store for statistics
        self.selections.append((self.game.leader, team)) 
        
        if leader == self:
            return
            
        leaderInfo = [p for p in self.playerInfos if p.player == leader][0]
        # store selected team for future. When we reveal spy we can process this data
        leaderInfo.onTeamSelected(team)
        
        if leaderInfo.isSpy():
            self._processSpySelections(leaderInfo)
            
    def vote(self, team): 
        
        if self.game.turn == 1: # always vote True in first turn
            return True

        # resistance has no reason to vote false, so don't reveal myself and vote always true
        if self.game.tries == 5:
            return True
        
        # num players in team equal to num of resistance and I'm not there -> one of them is spy    
        if len(self.game.team) == (self.playersCnt - self.spiesCnt) and not self in self.game.team:
            return False

        if self.spy:
            if len(self.game.team) == len([p for p in self.game.team if p in self.spies]):
                return False
        
            res = len([p for p in self.game.team if p in self.spies]) > 0
            return res

        suspicionLimit = self.playerInfos[self.playersCnt-self.spiesCnt-1].suspicion
        if set(team).intersection([p.player for p in self.playerInfos if p.suspicion >= suspicionLimit]):
            return False

        return True
        

    def onVoteComplete(self, votes):
        
        self.votes.append((self.game.turn, self.game.team[:], votes))
        self.lastVotes = votes
        
        # when all players are against the team, there is probably no spy in it
        # the probability is the lowest probability that spy votes for spy
        if not reduce(lambda x, y: x if x else y, votes, False):
            s = reduce(lambda x,y: self.playerStats[y.name].spyVotesForSpy.get() if self.playerStats[y.name].spyVotesForSpy .get() < x else x, self.game.team, 1.0)
            if s > 0.5: # only when spies votes for spies in more than half votings
                for pi in [pi for pi in self.playerInfos if pi.player in self.game.team]:
                    pi.onTrustedAction(s)
        
        # suspect players who vote against team in first turn
        if self.game.turn == 1:
            for pi in [pi for pi in self.playerInfos if not votes[pi.player.index]]:
                ratio = self._getSpyRatio(pi.player, 'spyVotesFalseFirstTurn', 'resVotesFalseFirstTurn')
                pi.onSuspiciousAction(ratio)
        else:
            # reduce suspicion for player who woted against the team when I'm against the team
            if not self.spy and not votes[self.index]:
                for pi in [pi for pi in self.playerInfos if not votes[pi.player.index]]:
                    pi.onTrustedAction(0.8) # TODO remove this hardcoded value
                    
            # when it's team with lenght equal to number of resistance, those who are not in it and vote for team may be spies
            if len(self.game.team) == (self.playersCnt - self.spiesCnt):
                for pi in [pi for pi in self.playerInfos if votes[pi.player.index] and pi.player not in self.game.team]:
                    ratio = self._getSpyRatio(pi.player, 'spyVotesForFullTeam', 'resVotesForFullTeam')
                    pi.onSuspiciousAction(ratio)
                    
        if self.game.tries == 5:
            # suspect players who wote against the team in last try
            for pi in [pi for pi in self.playerInfos if not votes[pi.player.index]]:
                pi.onSuspiciousAction(1.0) # TODO maybe should not be hardcoded
            
        self._updatePlayersSuspicions()
        
    
    def sabotage(self):
        if self.game.losses == 2 or self.game.wins == 2:
            return True

        #if self.game.turn == 1 or len(self.game.team) == 2:
        #   return False
    
        spiesOnMission = [s for s in self.spies if s in self.game.team]
        if len(spiesOnMission) == len(self.game.team):
            return False
    
        if len(spiesOnMission) > 1:
            p = sum([self.playerStats[s.name].sabotage.get() for s in spiesOnMission if s != self])/float(len(spiesOnMission)) 
            # randomly sabotage based on probability that other spy will sabotage. But we are sceptic, so lower the value little bit
            return random.random() >= (p*p)
        
        return True
        
    def onMissionComplete(self, sabotaged):
        
        self.missions.append((self.game.team[:], sabotaged))
        
        # trust players who didn't sabotage mission
        if sabotaged == 0:
            # dont trust too much to team that was approved by all players
            for pi in [pi for pi in self.playerInfos if pi.player in self.game.team]:
                if reduce(lambda x, y: x if not x else y, self.lastVotes, True):
                    susp = (self.playerStats[pi.player.name].sabotage.get()+0.5)/float(2)
                else:
                    susp = self.playerStats[pi.player.name].sabotage.get()
                pi.onTrustedAction(susp)
                
            # num players in team equal to num of resistance and mission passed. I'm almost sure that players not in mission are spies  
            if not self.spy and len(self.game.team) == (self.playersCnt - self.spiesCnt) and self in self.game.team:
                # probability that player which are not on missions are spies is equal to lowest probability of sabotage from team players
                susp = reduce(lambda x,y: self.playerStats[y.name].sabotage.get() if self.playerStats[y.name].sabotage.get() < x else x, self.game.team, 1.0)
                for pi in [pi for pi in self.playerInfos if pi.player not in self.game.team]:
                    pi.onSuspiciousAction(susp)
                    
                
        else:
            self.gameCombinations.addSabotage(self.game.team, sabotaged)
            
            # don't trust to players who woted for spies. Except first turn
            if self.game.turn != 1:
                for pi in [ pi for pi in self.playerInfos if self.lastVotes[pi.player.index]]:
                    pi.onSuspiciousAction(self._getSpyRatio(pi.player, 'spyVotesForSpy', 'resVotesForSpy'))
                            
            
        self._updatePlayersSuspicions()
        
    def onGameComplete(self, win, spies):
        # process all stored data to update statistics
        for team, sabotaged in self.missions:
            spiesOnMission = [p for p in team if p in spies]
            if len(spiesOnMission) > 0:
                for spy in spiesOnMission:
                    self.playerStats[spy.name].sabotage.add(float(sabotaged) / float(len(spiesOnMission)))
                    
        for turn, team, votes in self.votes:
            if turn == 1:
                for p in team:
                    if p in spies:
                        self.playerStats[p.name].spyVotesFalseFirstTurn.add(0 if votes[p.index] else 1)
                    else:
                        self.playerStats[p.name].resVotesFalseFirstTurn.add(0 if votes[p.index] else 1)
            else:
                if set(team).intersection(spies):
                    for p in self.game.players:
                        if p in spies:
                            self.playerStats[p.name].spyVotesForSpy.add(1 if votes[p.index] else 0)
                        else:
                            self.playerStats[p.name].resVotesForSpy.add(1 if votes[p.index] else 0)
                            
                if len(team) == (self.playersCnt - self.spiesCnt):
                    for p in [p for p in self.game.players if p not in team]:
                        if p in spies:
                            self.playerStats[p.name].spyVotesForFullTeam.add(1 if votes[p.index] else 0)
                        else:
                            self.playerStats[p.name].resVotesForFullTeam.add(1 if votes[p.index] else 0)
    
        for leader, team in self.selections:
            isSpyInTeam = len([p for p in team if p in spies and p != leader]) > 0
            if leader in spies:
                self.playerStats[p.name].spySelectsOtherSpy.add(1 if isSpyInTeam else 0)
    

    def _updatePlayersSuspicions(self):
        spyRatios = self.gameCombinations.getProbabilities()    
        for p in self.playerInfos:
            spyRatio = spyRatios[p.player.index]
            if spyRatio > 0.99:
                self._processSpySelections(p)
            p.updateSuspicion(spyRatio)
                
        self.playerInfos.sort(key=lambda x: x.suspicion)    

    def _getSpyRatio(self, player, spyAttr, resAttr):
        s = self.playerStats[player.name].__dict__[spyAttr].get()
        r = self.playerStats[player.name].__dict__[resAttr].get()
        return 0 if (s+r) == 0 else (s / (s + r))
        
    def _processSpySelections(self, leaderInfo):
        if len(leaderInfo.selectedPlayers) == 0: 
            return
            
        ratio = self.playerStats[leaderInfo.player.name].spySelectsOtherSpy.get()
        for p in self.playerInfos:
            if p in leaderInfo.selectedPlayers:
                p.onSuspiciousAction(ratio)
            else:
                p.onTrustedAction(ratio)
    
        leaderInfo.selectedPlayers = []


def intersection(*lists):
    """ Return the intersection of all of the lists presented as arguments.
    """
    if len(lists) == 0: return []
    ilist = lists[0]
    for i in range (1,len(lists)):
        ilist = [x for x in ilist if x in lists[i]]
    return ilist

def nonempty_intersection(*lists):
    """ Find the first list that is nonempty (lists[i]). 
        Then find the intersection(lists[i],lists[i+1],...) sequentially,
        but ignore lists whose intersection with previous intersections leads to an empty set,
    """
    i = 0
    while i < len(lists) and len(lists[i]) == 0: i += 1
    if i == len(lists): return []

    nelist = lists[i]
    for j in range (i+1, len(lists)):
        templist = [x for x in nelist if x in lists[j]]
        if templist != []: nelist = templist
    return nelist

class Rebounder(Bot):

    def onGameRevealed(self, players, spies):
        self.spies = spies
        # logging.debug("Revealed: " + str(players))

        if self.spy: 
            self.strong_configs = self.weak_configs = self.vote_configs = self.select_configs = [] # modelling opponents irrelevant for spies
        else: 
            # each of these configs corresponds to a possible arrangement of spies among the other players
            # strong_configs is a lower bound - it assumes spies always defect
            # weak_configs is an upper bound (and always true) - it assumes spies may or may not defect
            # vote_configs assumes a spy will vote against any team that does not contain a spy
            # select_configs assumes a spy will always choose a team consisting of himself plus non-spies
            self.strong_configs = range(6)
            self.weak_configs = range(6)
            self.vote_configs = range(6)
            self.select_configs = range(6)

    def configs_to_string(self, configs):
        """ Return the possible spy pairs for the given list of configs (from the point of view of self) 
        """
        outstr = ""
        for c in configs:
            for s in self.get_spies(c):
                outstr += str(s)
            outstr += " "
        return outstr

    def get_spies(self, v):
        """ Get the list of spy indices corresponding to config[v]
            Never includes me. Not used if I am a spy.
        """
        notme = [p for p in range(5) if p != self.index]
        if v==0:   return [notme[0], notme[1]]
        elif v==1: return [notme[0], notme[2]]
        elif v==2: return [notme[0], notme[3]]
        elif v==3: return [notme[1], notme[2]]
        elif v==4: return [notme[1], notme[3]]
        elif v==5: return [notme[2], notme[3]]
        else: assert(False)

    def get_resistance(self,v):
        """ Get the list of resistance indices corresponding to config[v]. 
            Never includes me. Not used if I am a spy.
        """
        notme = [p for p in range(5) if p != self.index]
        return [p for p in notme if p not in self.get_spies(v)]

    def select(self, players, count):
        """ Select players for a mission I will lead.
        """
        if self.spy:
            # return me plus (count-1) non-spies
            resistance = [p for p in players if p != self and p not in self.spies]
            return [self] + random.sample(resistance, count-1)
                #random.sample([p for p in players if p.index != self.index], count-1)
        else:
            assert len(self.get_select_configs()) > 0
            indices = random.sample(self.get_resistance(random.choice(self.get_select_configs())), count-1) + [self.index]
            return [p for p in players if p.index in indices]

    def get_select_configs(self):
        return nonempty_intersection(self.weak_configs, self.strong_configs, self.select_configs, self.vote_configs)

    def vote(self, team):
        """ Vote for a mission team proposed by leader (given a number of tries to date).
        """
        if self.spy:
            if self.game.tries == 5:
                return False
            else:
                return len([p for p in team if p in self.spies]) > 0 # Vote yes if spy on team, no otherwise
        else:
            if self.game.leader.index == self.index: return True
            if self.game.tries == 5:
                return True
            self.select_configs = [c for c in self.select_configs if self.select_compatible(c,team,self.game.leader)] 
            ti = [p.index for p in team]
            # Is the team compatible with a config which contains no spy?
            if len(self.strong_configs) == 0:
                vote = len([c for c in self.weak_configs if self.compatible(c,ti,0,False)]) > 0
            else:
                vote = len([c for c in self.strong_configs if self.compatible(c,ti,0,True)]) > 0
            if vote: votestring = "for"
            else: votestring = "against" 
            # logging.debug("Me: " + str(self.index) + ". Try: " + str(self.game.tries) + ". Vote " + votestring + " Team: " + str(ti) \
            #    + ". Possible strong configs: " + self.configs_to_string([c for c in self.strong_configs if self.compatible(c,ti,0,True)]) \
            #    + ". Possible weak configs: " + self.configs_to_string([c for c in self.weak_configs if self.compatible(c,ti,0,False)]))
            return vote

    def onVoteComplete(self, votes):
        self.vote_configs = [c for c in self.vote_configs if self.vote_compatible(c,self.game.players,votes,self.game.team)]
        # if not self.spy: logging.debug("Votes: " + str(votes))

    def vote_compatible(self, config, players, votes, team):
        """ Is config compatible with the voting assuming a spy will vote against any team that does not contain a spy?
            Also assume that all spies will vote against any team on the 4th try and any resistace will vote for.
        """
        forids = [players[i].index for i in range(5) if votes[i] == True] 
        num_selected_spies = len([s for s in team if s.index in self.get_spies(config)])
        num_spyvotesfor = len([p for p in self.get_spies(config) if p in forids])
        return (num_selected_spies > 0 or self.game.tries == 5) and num_spyvotesfor == 2 \
            or num_selected_spies == 0 and num_spyvotesfor == 0

    def select_compatible(self, config, team, leader):
        """ Is config compatible with the selection assuming a spy will 
            always propose a team consisting of himself plus non-spies
        """
        num_selected_spies = len([s for s in team if s.index in self.get_spies(config)])
        return leader.index in [p.index for p in team] and \
            leader.index in self.get_spies(config) and \
            num_selected_spies == 1

    def compatible(self, config, selectedplayerindices, sabotage, strong):
        """ Is config compatible with the mission result assuming spies always lie (strong == True)
            or assuming spies may or may not lie (strong == False)
        """
        num_selected_spies = len([s for s in selectedplayerindices if s in self.get_spies(config)])
        if strong: 
            return num_selected_spies == sabotage
        else:
            return num_selected_spies >= sabotage            

    def onMissionComplete(self, sabotaged):
        """ Update legal strong and weak configs given a mission result.
        """
        if not self.spy:
            spi = [p.index for p in self.game.team]
            self.strong_configs = [c for c in self.strong_configs if self.compatible(c,spi,sabotaged,True)] 
            self.weak_configs = [c for c in self.weak_configs if self.compatible(c,spi,sabotaged,False)] 
            # logging.debug("Me: " + str(self.index) + ". Mission result: " + str(sabotaged) + " for team " + str(spi) +\
            #    ". Possible strong configs: " + self.configs_to_string(self.strong_configs) + \
            #    ". Possible weak configs: " + self.configs_to_string(self.weak_configs) + \
            #    ". Possible vote configs: " + self.configs_to_string(self.vote_configs) + \
            #    ". Possible select configs: " + self.configs_to_string(self.select_configs) + "\n")

    def sabotage(self):
        """ Sabotage if I am the only spy in the team, or if the other spy on the team is random,
            or if it is two of me as spies and I have the lower index.
        """
        if self.spy:
            other_spy = [s for s in self.spies if s.index != self.index][0]
            if other_spy.index not in [p.index for p in self.game.team] or \
                other_spy.name == "Random" or \
                (other_spy.name == self.name and self.index < other_spy.index):
                return True
        return False

    def onGameComplete(self, players, spies):
        # logging.debug("-----------------------------------------------\n")
        pass


def permutations(config):
    """Returns unique elements from a list of permutations."""
    return list(set(itertools.permutations(config)))


class PandSBot(Bot):

    def onGameRevealed(self, players, spies):
        self.players = players
        self.spies = spies
        self.team = None        
        self.friends = [[Probability(0.4) for x in range(5)] for y in range(5)]
        self.suspects = [Probability(0.0) for x in range(5)]
        self.supportSuspects = [Variable(0.4, 1.0) for x in range(5)]
        self.suspeciousActions = [Variable(0,0) for x in range(5)] #player not in team, team == 3, votes
        self.possibleGoodActions = [Variable(0,0) for x in range(5)] #player in team, votes against team
        self.suspectsPair = [[(x,y),0] for x in range(5) for y in range(5) if x < y]

    def _updateSuspectsPair(self):

        for x in self.suspectsPair:
            spy1 = x[0][0]
            spy2 = x[0][1]
            #calculate how suspicious x[0] pair (spy1 friend for spy2, spy2 friend for spy1, and etc)
            estimate = self.suspects[spy1].estimate() * self.suspects[spy2].estimate();
            if estimate < 0.99:                 
                v = (0.50 + 0.50 * self.friends[spy1][spy2].estimate() * self.friends[spy2][spy1].estimate())
                v *= (0.75 + 0.25 * self.supportSuspects[spy1].estimate() * self.supportSuspects[spy2].estimate())
                v *= estimate
                v *= 0.4 + 0.6 * (self.suspeciousActions[spy1].estimate() + self.suspeciousActions[spy2].estimate())/2
                v *= 1 - 0.1 * (self.possibleGoodActions[spy1].estimate() + self.possibleGoodActions[spy2].estimate())/2
                x[1] = v
                #x[1] =(random.uniform(0.98, 1.0))*x[1]
            else:
                x[1] = estimate

    def _getSuspicious(self, spy1):
        v = (0.75 + 0.25 * self.supportSuspects[spy1].estimate())
        v *= self.suspects[spy1].estimate()
        v *= 0.4 + 0.6 * (self.suspeciousActions[spy1].estimate())
        v *= 1 - 0.1 * (self.possibleGoodActions[spy1].estimate())
        return v

    def _getBadPair(self):
        #get the most suspicious pair
        tmp = [x for x in self.suspectsPair if self.index not in x[0]]
        result = max(tmp, key=lambda p: (random.uniform(0.9, 1.0))*p[1])
        #result = tmp
        if result[0] > 0:#random.uniform(0, 0.5):-------------------------------------------
            return result[0],result[1]
        else:
            return []

    def _getGood(self):
        #get all players that is not in badPair
        bad, v = self._getBadPair()
        if v > 0:
            t = set(self._othersIndexes())-set(bad)
            result = sorted(t, key=lambda p: self._getSuspicious(p))
        else:
            result = sorted(self._othersIndexes(), key=lambda p: (p - self.game.leader.index + len(self.game.players))%len(self.game.players))
            
        return result


    def _othersIndexes(self):
        #all players indexes [1,2,...] except users
        return [x.index for x in self.others() ]

    def _playersIndexes(self):
        #all players indexes [1,2,...]
        return [x.index for x in self.game.players ]

    def _maybeLastTurn(self):
        return (self.game.losses == 2) or (self.game.wins == 2)

    def select(self, players, count):
        
        goodPlayers = self._getGood();
        badPair, v = self._getBadPair();

        #always include myself
        me = [p for p in players if p.index == self.index]

        
        #other variants from reliable players
        #variants = [p for p in players if p.index in goodPair and p not in self.spies]
        #num = min(len(variants), count - 1)
        #result = list(set( me + random.sample(variants, num)))
        result = me + [x for x in players if x.index in goodPlayers[0: count-1]]

        #if not enough add random
        if (len(result)<count):
            #self._getSortedGood()
            if not self.spy:
                self.log.info("get random=(")
            others = [p for p in players if p not in result]
            result += random.sample(others, count-len(result))
        return result

    def vote(self, team): 
        othersPeople = set(self.game.players)-set(team);

        #always vote for own team
        if self.game.leader == self:
            return True
        if self.game.turn==1 and self.game.tries == 1:
            return True
        
        badPair,v = self._getBadPair()

        teamIndexes = set([x.index for x in team])

        #As a spy, vote for all missions that include one spy!
        if self.spy:            
            spiesCount = len([p for p in team if p in self.spies])
            if spiesCount > 0: # or (spiesCount > 0 and self._maybeLastTurn()):
                return True
            return False

        # As resistance, always pass the fifth try.
        if self.game.tries == 5:
            return True
        # If I'm not on the team and it's a team of 3!
        if len(team) == 3 and not self.index in [p.index for p in team]:
            return False

        ## first round vote, because we do not have any information
        #if self.game.losses == 0:
        #    return True
        #do not vote for most suspicious pair
        if teamIndexes.intersection(badPair):
            return False
        if self.game.leader.index not in teamIndexes:
            return False  
        # Otherwise, just approve the team and get more information. 
        return True

    def onVoteComplete(self, votes):
        me = [p for p in self.game.players if p.index == self.index]
        votes = votes
        self.votes = votes;#to work properly votes[p.index]
        
        team = [p.index for p in set(self.game.team)-set(me)]
        notTeam = [p.index for p in set(self.game.players)-set(self.game.team)-set(me)];

        #leader didn't choose himself
        self.suspeciousActions[self.game.leader.index].sampleBool(self.game.leader.index not in team)

        for p1 in self._othersIndexes():

            # if 1st round, 1st try and player against - suspicious ----------------------------------
            self.suspeciousActions[p1].sampleBool( not self.votes[p1] and self.game.turn==1 and self.game.tries==1)

            # if 5th tries and player against - suspicious ----------------------------------
            self.suspeciousActions[p1].sampleBool( not self.votes[p1] and self.game.tries==5)

                
            #player out of team of 3 person, but vote, maybe he is spy (or stupid)
            self.suspeciousActions[p1].sampleBool( self.votes[p1] and len(self.game.team)==3 and p1 not in team)

            #player in team, but votes againts, possible he is good (or stupid=))
            self.possibleGoodActions[p1].sampleBool( not self.votes[p1] and p1 in team)

            if p1 == self.game.leader.index:
                #spy doesnot choose second spy in team
                for p2 in notTeam:
                    self.friends[p1][p2].sampleExt(1, len(notTeam))
            else:
                #anyone vote for team where he is, so more intrested in team without
                if p1 not in team:
                    # for all players that voted,  team are possible friends
                    if (self.votes[p1]):                        
                        for p2 in team:
                            self.friends[p1][p2].sampleExt(1, len(team))
                    else:
                        for p2 in notTeam:
                            self.friends[p1][p2].sampleExt(1, len(notTeam))                    
                    
        self._updateSuspectsPair()        
        self.team = None
    
    def _getIndexes(self, players):
        return [p.index for p in players]

    def onMissionComplete(self, sabotaged):

        #update possibility to be a spy
        maxSpies = 2

        team = [p.index for p in self.game.team if p.index != self.index]
        other = [p.index for p in self.game.players if p.index != self.index and p.index not in team]

        if sabotaged>0:
            for i in team:    
                self.suspects[i].sampleExt(sabotaged, len(team))

        if sabotaged<maxSpies:
            for i in other:    
                self.suspects[i].sampleExt(maxSpies-sabotaged, len(other))
      
        if self.game.turn > 1:
            for p in other:#[p.index for p in self.game.players if p.index != self.index]:
                val = int((self.votes[p] and sabotaged>0) or (not self.votes[p] and sabotaged==0))
                self.supportSuspects[p].sampleExt(val,1)

        self._updateSuspectsPair()        

    def sabotage(self):
        #return self.spy
        #sabotage only if one spy or last turn
        spiesCount = len([p for p in self.game.team if p in self.spies])
        return spiesCount == 1 or self._maybeLastTurn() or (spiesCount==2 and self.game.leader.index != self.index)
    
    def onGameComplete(self, win, spies):
        pass
        #if self.spy:
        #    return
        #if not win:
        #    opinion = (self._getBadPair())
        #    good = len(set(opinion) - set([p.index for p in spies]))==0;
        #    if not good:
        #        print "Fail=("
        #        print "Turn %s"%self.game.turn
        #        print "%s (%s)"%( good, spies)
        #        for pair in self.suspectsPair:
        #            if self.index not in pair[0]:
        #                s1 = pair[0][0]
        #                s2 = pair[0][1]
        #                print "%s : %0.2f (%s * %s)=%0.2f; (%s %s) (%s %s)"%((s1, s2), pair[1], self.friends[s1][s2], self.friends[s2][s1], self.friends[s1][s2].estimate() * self.friends[s2][s1].estimate() ,self.supportSuspects[s1],self.supportSuspects[s2], self.suspects[s1], self.suspects[s2])
        #        print "Suspects:"
        #        print self.suspects
        #        print "Support:"
        #        print self.supportSuspects
        #        print "Friends:"
        #        for x in self.friends:
        #            print x
        #        print "PossibleGoodAction: "
        #        print self.possibleGoodActions
        #        print "Suspicious Action:"
        #        print self.suspeciousActions

        #        #print "Friends delta:"
        #        #for x in self.friendsExclusive:
        #        #    print ["%0.2f%%"%(f*100) for f in x]
        #        print self.game.players
        #        print "(%s, %s)"%(opinion[0], opinion[1])
        #        pass
        #    else:
        #        print "OK=)"
        #else:
        #    print "OK=)"



class Variable(object):
    def __init__(self, v0, n0):
        self.total = v0
        self.samples = n0

    def sample(self, value):
        self.sampleExt(value, 1)

    def sampleBool(self, value):
        self.sampleExt(int(value), 1)

    def sampleExt(self, value, n):
        self.total += value
        self.samples += n

    def estimate(self):
        if self.samples > 0:
            return float(self.total) / float(self.samples)
        else:
            return 0.0
    def error(self):
            # We're dealing with potential results at 0% or 100%, so
            # use an Agresti-Coull interval (can't do a normal
            # Clopper-Pearson without requiring the numpy library, and
            # can't use the central limit theorem too close to the
            # extremes). Note this is still degenerate when the number
            # of samples is very small, and may give an upper bound >
            # 100%.
            n_prime = self.samples + 3.84 # 95% confidence interval
            value = (self.total + (3.84 * 0.5)) / n_prime
            error = 1.96 * math.sqrt(value * (1.0 - value) / n_prime)
            return error

    def estimateWithError(self):
        return self.estimate()-self.error()*0.25

    def estimateWithErrorRnd(self):
        val = self.estimate()-self.error()*random.uniform(0.0, 0.2)
        val = max(0, min(val,1))
        return val

    def __repr__(self):
        if self.samples:
            #return "%0.2f%% (%i)" % ((100.0 * float(self.total) / float(self.samples)), self.samples)
            return "%0.2f%% " % ((100.0 * float(self.total) / float(self.samples)))
        else:
            return "UNKNOWN"


class Probability(object):
    def __init__(self, v0):
        self.value = v0
        self.n = 0

    def sample(self, value):
        self.sampleExt(value, 1)

    def sampleExt(self, value, n):
        self.value = 1 - (1 - self.value)*(1 - float(value) / float(n))
        self.n += n
        
    def sampleExtNeg(self, value, n):
        self.value *= (1- float(value) / float(n))

    def estimate(self):
        return self.value

    def __repr__(self):
        #return "%0.2f%% (%i)" % (100.0 * float(self.value), self.n)
        return "%0.2f%% " % (100.0 * float(self.value))


class Opeth(Bot):
    """Opeth is a melancholic bot."""

    # List of spies. We only get this info if we are a spy.
    spy_spies = None

    # My interpretation of the world. This is dict where keys are Player objects
    # and values are confidence scores for each player. Confidence scores start
    # at 0 and get higher if I believe that player is not a spy.
    # We do not use any additional info granted to us if we are a spy to update
    # this.
    my_guess = dict()

    # This is like my_guess except we update it based on information that other
    # resistance members have. It is not as accurate as my_guess but will
    # provide an insight on what other bots think about the world.
    their_guess = dict()

    # List of players that I am sure are spies.
    spies_for_sure = set()

    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players  List of all players in the game including you.
        @param spies    List of players that are spies, or an empty list.
        """
        self.spy_spies = spies

        self.my_guess = dict(zip(players, [0] * 5))
        self.their_guess = dict(zip(players, [0] * 5))

    def onMissionAttempt(self, mission, tries, leader):
        """Callback function when a new turn begins, before the
        players are selected.
        @param mission  Integer representing the mission number (1..5).
        @param tries    Integer count for its number of tries (1..5).
        @param leader   A Player representing who's in charge.
        """
        pass

    def select(self, players, count):
        """Pick a sub-group of players to go on the next mission.
        @param players  The list of all players in the game to pick from.
        @param count    The number of players you must now select.
        @return list    The players selected for the upcoming mission.
        """
        me = [p for p in players if p.index == self.index]

        # As a spy, pick myself and others who are not spies.
        # if self.spy:
        #    others = [p for p in players if p not in self.spy_spies]
        #    return me + random.sample(others, count-1)

        # It makes a lot of sense to be part of the team so we will always propose ourselves.

        if not self.spy:
            # I am not a spy. Select myself and the people I trust the most.
            team = self.getPlayersITrust(count)
        else:
            # I am a spy. Select myself and the people I think they trust the most.
            team = self.getPlayersTheyTrust(count)

        # If I am not on that list, replace the last element
        if me[0] not in team:
            team.pop()
            team = team + me

        return team

    def onTeamSelected(self, leader, team):
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        @param leader   The leader in charge for this mission.
        @param team     The team that was selected by the current leader.
        """

        # Because we are bots and there is no conversation involved, there is no
        # point in not selecting yourself as part of the team. If they didn't
        # do it, that is suspicious.
        if leader not in team:
            self.my_guess[leader] -= 5
            self.their_guess[leader] -= 5

        pass

    def vote(self, team):
        """Given a selected team, decide whether the mission should proceed.
        @param team      List of players with index and name.
        @return bool     Answer Yes/No.
        """

        self.log.debug("Voting for:")
        self.log.debug(team)

        # Approve my own team.
        if self.index == self.game.leader.index:
            self.log.debug("Yup. I am the leader.")
            return True

        # Both types of factions have constant behavior on the last try.
        if self.game.tries == 5:
            self.log.debug("Last try!")
            return not self.spy

        # Spies select any mission with one or more spies on it.
        if self.spy:
            self.log.debug("Yup. I am a spy and there is at least one spy on the team.")
            return len([p for p in team if p in self.spy_spies]) > 0
    
        # I am resistance. Vote against teams that have at least one of 2 most untrustested players
        worst = self.getPlayersITrust(2, True)
        for player in worst:
            if player in team and self.my_guess[player] < 0:
                self.log.debug("Nope. I don't trust these guys:")
                self.log.debug(worst)
                self.log.debug(team)
                return False
    
        #If I'm not on the team, and it's a team of 3...
        if len(team) == 3 and not self in team:
            self.log.debug("Nope. It's a a team of 3 and I am not in it:")
            self.log.debug(team)
            return False    

        return True

    def onVoteComplete(self, votes):
        """Callback once the whole team has voted.
        @param votes        Boolean votes for each player (ordered).
        """
        if votes > 3:
            self.log.debug("* Team approved!")
        else:
            self.log.debug("* Team rejected!")
        pass

    def sabotage(self):
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        @return bool        Yes to shoot down a mission.
        """

        spies = [s for s in self.game.team if s in self.spy_spies]

        # Special cases:

        # Sabotage to win (spies won 2 times already)
        if self.game.losses == 2:
            self.log.debug("Sabotaging because spies have won 2 missions")
            return True

        # Sabotage to not loose (resistance won 2 times already). 
        # Only do it if there is at least a non spy on the team.
        if self.game.wins == 2 and len(spies) < len(self.game.team):
            self.log.debug("Sabotaging because resistance has won 2 missions")
            return True

        # If I am the only spy
        if len(spies) == 1:
            return True
            if (len(self.game.team) == 3):
                self.log.debug("Sabotaging because I am the only spy on a team of 3.")
                return True

            if (len(self.game.team) == 2):
                # We will confuse them only if they haven't won anything.
                self.log.debug("Sabotaging: " + str(self.game.wins != 0) + " because I am the only spy and the number of wins is " + str(self.game.wins))
                return self.game.wins != 0
            return True

        if len(spies) > 1:
            if self.index == self.game.leader.index:
                self.log.debug("Sabotaging. There is more than one spy in the team but I am the leader")
                return True

            if self.game.leader in spies:
                self.log.debug("Not sabotaging. There is more than one spy but I am not the leader.")
                return False

            # More than one spy and non of us is the leader. 
            # Make this decision based on the number of wins.
            self.log.debug("Sabotaging: " + str(self.game.wins > 1) + " because I am not the only spy and the number of wins is " + str(self.game.wins))
            return self.game.wins > 1

        return True

    def onMissionComplete(self, sabotaged):
        """Callback once the players have been chosen.
        @param selected     List of players that participated in the mission.
        @param sabotaged    Integer how many times the mission was sabotaged.
        """

        # Can we know for sure if the whole team are spies?
        if len(self.game.team) == sabotaged:
            for spy in self.game.team:
                if spy.index != self.index:
                    self.my_guess[spy] -= 100
                    self.spies_for_sure.add(spy)
                self.their_guess[spy] -= 100

        # Can we know for sure if the rest of the team is a spy?
        # 3 conditions: I am not a spy, I am in the team,
        # and the number of sabotaged votes is equal to the
        # team size minus one (my vote)
        if not self.spy and self in self.game.team and sabotaged == len(self.game.team) - 1:
            for spy in self.game.team:
                if spy.index != self.index:
                    self.my_guess[spy] -= 100
                    self.spies_for_sure.add(spy)

        # If this mission failed, that team gets penalized (according to the number of times the mission was sabotaged),
        # otherwise we gain confidence in them.
        for player in self.game.team:
            if sabotaged:
                if player.index != self.index:
                    self.my_guess[player] -= sabotaged
                self.their_guess[player] -= sabotaged
            else:
                self.my_guess[player] += 2
                self.their_guess[player] += 2

        # If the leader is not in the team but the mission succeded, restore confidence in him, I guess.
        if self.game.leader not in self.game.team:
            if sabotaged == 0:
                self.my_guess[self.game.leader] += 5
                self.their_guess[self.game.leader] += 5
            else:
                # If the mission was sabottaged, punish him further.
                self.my_guess[self.game.leader] -= sabotaged
                self.their_guess[self.game.leader] -= sabotaged

        # Print stats
        if sabotaged == 0:
            self.log.debug("*** SUCCEDED ***")
        else:
            self.log.debug("*** SABOTAGED *** " + str(sabotaged) + " times.")
        self.log.debug("Leader: " + self.game.leader.name);

        self.log.debug("--- Team:")
        for player in self.game.team:
            self.log.debug(player.name + ": " +  str(self.my_guess[player]))

        self.log.debug("--- The rest:")
        for player in self.game.players:
            if player in self.game.team: 
                continue
            self.log.debug(player.name + ": " +  str(self.my_guess[player]))


        self.log.debug("---------------------------------------------------")


    def onGameComplete(self, win, spies):
        """Callback once the game is complete, and everything is revealed.
        @param win          Boolean if the Resistance won.
        @param spies        List of only the spies in the game.
        """

        self.log.debug("*************************** GAME RESULTS ***************************")
        self.log.debug("Am I as spy? " + str(self.spy))
        if self.spy:
            self.log.debug("--- Actual spies:")
            for player in self.spy_spies:
                self.log.debug(player.name)
        self.log.debug("--- My guess:")
        for player in self.game.players:
            self.log.debug(player.name + ": " +  str(self.my_guess[player]))
        self.log.debug("--- What I think is their guess:")
        for player in self.game.players:
            self.log.debug(player.name + ": " +  str(self.my_guess[player]))
        self.log.debug("******************************* DONE *******************************")
        self.log.debug(self.my_guess)

        pass

    def getPlayersITrust(self, number_players, reversed=False):
        """Returns a sorted list of the number_players more trustworthy players.
        @param number_players How many players do you want?
        """
        sorted_list = sorted(self.game.players, key=lambda player: self.my_guess[player], reverse=not reversed)
        sorted_list.remove(self)
        return sorted_list[:number_players]

    def getPlayersTheyTrust(self, number_players, reversed=False):
        sorted_list = sorted(self.game.players, key=lambda player: self.their_guess[player], reverse=not reversed)
        return sorted_list[:number_players]


class MissionedTeam:
    
    def __init__(self,team,sabotages):
        self.team=team
        self.sabotages=sabotages


class Statistic:
    def __init__(self,defaultVal=0.0,minSucesos=1):
        self.ocurrences=0.0
        self.totalSucesos=0.0
        self.defaultVal=defaultVal
        self.minSucesos=minSucesos
    def ocurrence(self):
        self.ocurrences=self.ocurrences+1
        self.totalSucesos+=1
    def notOcurrence(self):
        self.totalSucesos+=1
    def update(self,ocurence):
        if ocurence:
            self.ocurrence()
        else:
            self.notOcurrence()
    def probability(self):
        if self.totalSucesos>=self.minSucesos:
            return self.ocurrences/self.totalSucesos
        else:
            return self.defaultVal
    def hasEnoughtSamples(self,minSucesos):
        return self.totalSucesos>=minSucesos
    def copy(self):
        myCopy=Statistic()
        myCopy.ocurrences=self.ocurrences
        myCopy.totalSucesos=self.totalSucesos
        myCopy.defaultVal=self.defaultVal
        myCopy.minSucesos=self.minSucesos
        return myCopy    
                
class PlayerStats:
    def __init__(self):
        self.playersStats={}
    def registerPlayer(self,player):
        if player.name not in self.playersStats:
            self.playersStats[player.name]={}
    def ocurence(self,suceso,player,state):
        self.update(suceso, player, True, state)
    def notOcurrence(self,suceso,player,state):
        self.update(suceso, player, False, state)
    def probabilityInternal(self,suceso,player):
        self.registerPlayer(player)
        playerStats=self.playersStats[player.name]
        if suceso not in playerStats:
            playerStats[suceso]=Statistic()
        return playerStats[suceso].probability()
    def enoughtData(self,suceso,player,minSamples):
        self.registerPlayer(player)
        playerStats=self.playersStats[player.name]
        if suceso not in playerStats:
            playerStats[suceso]=Statistic()
        return playerStats[suceso].hasEnoughtSamples(minSamples)    
    def probability(self,suceso,player,defaultValue=0.0,minSamples=1,hierachicalProbability=False,gameState=None):
        if gameState is not None and hierachicalProbability:
            if self.enoughtData(suceso+str(gameState.turn)+"-"+str(gameState.tries), player,minSamples):
                return self.probabilityInternal(suceso+str(gameState.turn)+"-"+str(gameState.tries), player)
            elif self.enoughtData(suceso+str(gameState.turn), player,minSamples):
                return self.probabilityInternal(suceso+str(gameState.turn), player)
        if self.enoughtData(suceso, player,minSamples):
            return self.probabilityInternal(suceso, player) 
        else:
            return defaultValue
    def update(self,suceso,player,ocurrence,gameState=None):
        self.updInternal(suceso, player, ocurrence)
        if gameState is not None:
            self.updInternal(suceso+str(gameState.turn), player, ocurrence)
            self.updInternal(suceso+str(gameState.turn)+"-"+str(gameState.tries), player, ocurrence)
            #self.updInternal(suceso+str(gameState.leader), player, ocurrence)
            
    def updInternal(self,suceso,player,ocurrence):
        self.registerPlayer(player)
        playerStats=self.playersStats[player.name]
        if suceso not in playerStats:
            playerStats[suceso]=Statistic()
        playerStats[suceso].update(ocurrence)        
    def copy(self):
        myCopy=PlayerStats()
        for playerName in self.playersStats:
            myCopy.playersStats[playerName]={}
            for suceso in self.playersStats[playerName]:
                myCopy.playersStats[playerName][suceso]=self.playersStats[playerName][suceso].copy()
        return myCopy

class GameState:
    
    def __init__(self,players):
        self.leader=None
        self.tries=1
        self.turn=1
        self.players=players
        self.votes=[]
        self.selectedTeams=[]
        self.lostTurns=0
        self.team=[]
        self.missionedTeams=[]
        self.selectionResults=[]
    def voteComplete(self,votes):
        self.selectedTeams.append(self.team)
        self.votes.append(votes)
#["Sabotear","VotarAFavorDeUnSpia","VotarEnContraDeUnSpia","SeleccionarSaboteo","SeleccionarOk","NoSabotear"]
class PlayerAsignment:
    
    
    def __init__(self,resistance,spies):
        self.resistance=resistance
        self.spies=spies
        self.playersStats={}
        self.sabotages=[]
        self.isValid=False
        #there are 6 posible player asignments given the fact that i'm from resistance
        self.probability=1.0/6.0
    def isPosible(self):
        return self.probability>0
    def alwaysTrue(self):
        return self.probability==1

class Rule:
    def __init__(self,spyStats,rsStats):
        self.spyStats=spyStats
        self.rsStats=rsStats
        self.init()
        pass
    def init(self):
        pass
    def applies(self,state,action,data,asignments):
        return False
    def updateModel(self,asignments,state, action,data):
        if self.applies(state, action, data,asignments):
            return self.upd(asignments,state, action,data)
            
    def upd(self,asignments,state, action,data):
        pass
    def spyProbability(self,asignments,p):
        pSpy=0
        for a in asignments:
            if p in a.spies:
                pSpy+=a.probability
        return pSpy
        
class Actions:
    SABOTAGE="Sabotage"
    VOTE="Vote"
    ACCOMPLISHED="Acomplished"
    SELECT="Select"

class Probabilities:
    VOTE_ON_TEAMS_WITH_SPIES="VoteOnTeamWithSpies"
    SELECT_SPY_TEAMS="SelectSpyTeams"
    VOTE_ON_HE="VoteOnHe"
    VOTE_ON_OTHER_SPIES="VOTE_ON_OTHER_SPIES"
    VOTE_ON_SPIES="VOTE_ON_SPIES"
    VOTE_ON_RESISTENCE="VoteSuccessfull"
    VOTE_ON_TWOSPY_TEAM="VoteOnTwoSpyTeam"
    SABOTAGE="Sabotage"
    SELECT_HIMSELF="SELECTHIMSELF"
    SELECT_SPIES="SELECT_SPIES"
    SABOTAGE_ON_TWO_SPIE_TEAM="SABOTAGE_ON_TWO_SPIE_TEAM"
    VOTE_ON_THREETEAMS_WHEREHE_ISNT="VOTE_ON_THREETEAMS_WHEREHE_ISNT"

        

class SabotageRule(Rule):
    def applies(self, state, action,data,asignments):
        return action==Actions.SABOTAGE and data>=0
    def upd(self, asignments, state, action, sabotages):
        probSum=0
        for a in asignments:
            spiesOnTeam=0
            for p in state.team:
                if p in a.spies:
                    spiesOnTeam=spiesOnTeam+1
            if spiesOnTeam<sabotages:
                a.probability = 0
            if spiesOnTeam==2 and sabotages==1:
                a.probability=min(a.probability,
                                  (self.spyStats.probability(Probabilities.SABOTAGE_ON_TWO_SPIE_TEAM,a.spies[1],0.0,5)+
                                  self.spyStats.probability(Probabilities.SABOTAGE_ON_TWO_SPIE_TEAM,a.spies[0],0.0,5))*0.5)    
            probSum += a.probability
        
        if probSum>0:
            adjustCoef=1/probSum
            for a in asignments:
                a.probability=a.probability*adjustCoef
        return state.team


class VoteAgainsMissionLeader(Rule):
    def init(self):
        self.checkVotes=[]
    def applies(self, state, action, data,asignments):
        if action== Actions.SABOTAGE:
                self.checkVotes.append(state.team)
        if len(state.team)==2 and action==Actions.VOTE and not data.vote and state.lostTurns>0:
            if data.player in state.team and data.player!=state.leader:
                for t in self.checkVotes:
                    if data.player in t and state.leader in t:
                        return True
        
        return False
    def upd(self, asignments, state, action, data):
        pSpy=self.spyProbability(asignments, data.player)
        if pSpy>0 and pSpy<1:
            pRes=1-pSpy
            pVotoEnContraLuegoSaboteoSpia=0.1
            pVotoEnContraLuegoSaboteoRs=1.0
            denominator=(pSpy*(pVotoEnContraLuegoSaboteoSpia-pVotoEnContraLuegoSaboteoRs)+pVotoEnContraLuegoSaboteoRs)
            if denominator>0:
                newPSpy=(pVotoEnContraLuegoSaboteoSpia*pSpy)/denominator
                newPRs=1-newPSpy
                for a in asignments:
                    if data.player in a.spies:
                        a.probability=a.probability*(newPSpy/pSpy)
                    else:
                        a.probability=a.probability*(newPRs/pRes)

class AcomplishedRule(Rule):
    def applies(self, state, action,data,asignments):
        return action==Actions.ACCOMPLISHED
    def upd(self, asignments, state, action, sabotages):
        for p in state.team:
            pSpy=self.spyProbability(asignments, p)
            if pSpy>0 and pSpy<1:
                pRes=1-pSpy
                pNoSaboteoSiendoSpia=1-self.spyStats.probability(Probabilities.SABOTAGE,p,1.0,40)
                #assert pNoSaboteoSiendoSpia<0.5,p.name+str(pNoSaboteoSiendoSpia)
                pNoSaboteoSiendoResistencia=1.0
                denominator=(pRes*(pNoSaboteoSiendoResistencia-pNoSaboteoSiendoSpia)+pNoSaboteoSiendoSpia)
                if denominator>0:
                    newPRs=(pNoSaboteoSiendoResistencia*pRes)/denominator
                    newPSpy=1-newPRs
                    for a in asignments:
                        if p in a.resistance:
                            a.probability=a.probability*(newPRs/pRes)
                        else:
                            a.probability=a.probability*(newPSpy/pSpy)

class LastTryVote(Rule):
    def applies(self, state, action, data,asignments):
        return state.tries==5 and action==Actions.VOTE and not data.vote
    def upd(self, asignments, state, action, data):
        probSum=0
        for a in asignments:
            spiesOnTeam=0
            if data.player in a.spies:
                spiesOnTeam=spiesOnTeam+1
            if spiesOnTeam==0:
                a.probability = 0
            probSum += a.probability
        
        if probSum>0:
            adjustCoef=1/probSum
            for a in asignments:
                a.probability=a.probability*adjustCoef                

class NoVoteOnSuccessfullTeamTeams(Rule):
    def init(self):
        self.accomplishedPlayers=[]
    def applies(self, state, action, data,asignments):
        if action==Actions.ACCOMPLISHED:
            for p in state.team:
                self.accomplishedPlayers.append(p)
        elif action==Actions.SABOTAGE:
            for p in state.team:
                if p in self.accomplishedPlayers:
                    self.accomplishedPlayers.remove(p)
        if action==Actions.VOTE and not data.vote:
            for p in state.team:
                if p not in self.accomplishedPlayers:
                    return False
            return True
        else:
            return False
                
    def upd(self, asignments, state, action, data):
        pSpy=self.spyProbability(asignments, data.player)
        if pSpy>0 and pSpy<1:
            pRes=1-pSpy
            pVotoEnContraLuegoSaboteoSpia=1.0
            pVotoEnContraLuegoSaboteoRs=0.2
            denominator=(pSpy*(pVotoEnContraLuegoSaboteoSpia-pVotoEnContraLuegoSaboteoRs)+pVotoEnContraLuegoSaboteoRs)
            if denominator>0:
                newPSpy=(pVotoEnContraLuegoSaboteoSpia*pSpy)/denominator
                newPRs=1-newPSpy
                for a in asignments:
                    if data.player in a.spies:
                        a.probability=a.probability*(newPSpy/pSpy)
                    else:
                        a.probability=a.probability*(newPRs/pRes)         

class NoVoteOnTeamsOfThreeWhereHeIsNot(Rule):
    def applies(self, state, action, data,asignments):
        return action==Actions.VOTE and len(state.team)==3 and data.player not in state.team
    
    def upd(self, asignments, state, action, data):
        pSpy=self.spyProbability(asignments, data.player)
        if pSpy>0 and pSpy<1:
            pRes=1-pSpy
            pVotoEnContraLuegoSaboteoSpia=self.spyStats.probability(Probabilities.VOTE_ON_THREETEAMS_WHEREHE_ISNT,data.player,0.9,5)
            pVotoEnContraLuegoSaboteoRs=self.rsStats.probability(Probabilities.VOTE_ON_THREETEAMS_WHEREHE_ISNT,data.player,0.0,5)
            if not data.vote:
                pVotoEnContraLuegoSaboteoSpia=1-self.spyStats.probability(Probabilities.VOTE_ON_THREETEAMS_WHEREHE_ISNT,data.player,0.9,5)
                pVotoEnContraLuegoSaboteoRs=1-self.rsStats.probability(Probabilities.VOTE_ON_THREETEAMS_WHEREHE_ISNT,data.player,0.0,5)
                
            denominator=(pSpy*(pVotoEnContraLuegoSaboteoSpia-pVotoEnContraLuegoSaboteoRs)+pVotoEnContraLuegoSaboteoRs)
            if denominator>0:
                newPSpy=(pVotoEnContraLuegoSaboteoSpia*pSpy)/denominator
                newPRs=1-newPSpy
                for a in asignments:
                    if data.player in a.spies:
                        a.probability=a.probability*(newPSpy/pSpy)
                    else:
                        a.probability=a.probability*(newPRs/pRes)     
class VoteOnSpies(Rule):
    def init(self):
        self.accomplishedPlayers=[]
    def applies(self, state, action, data, asignments):
        if action==Actions.VOTE and data.vote and state.tries<4:
            for p in state.team:
                pSpy=self.spyProbability(asignments, p)
                if pSpy>0.8:
                    return True
            return False
        else:
            return False
                
    def upd(self, asignments, state, action, data):
        pSpy=self.spyProbability(asignments, data.player)
        if pSpy>0 and pSpy<1:
            pRes=1-pSpy
            pVotoEnContraLuegoSaboteoSpia=self.spyStats.probability(Probabilities.VOTE_ON_SPIES,data.player,1.0,50)
            pVotoEnContraLuegoSaboteoRs=self.rsStats.probability(Probabilities.VOTE_ON_SPIES,data.player,0.1,50)
            denominator=(pSpy*(pVotoEnContraLuegoSaboteoSpia-pVotoEnContraLuegoSaboteoRs)+pVotoEnContraLuegoSaboteoRs)
            if denominator>0:
                newPSpy=(pVotoEnContraLuegoSaboteoSpia*pSpy)/denominator
                newPRs=1-newPSpy
                for a in asignments:
                    if data.player in a.spies:
                        a.probability=a.probability*(newPSpy/pSpy)
                    else:
                        a.probability=a.probability*(newPRs/pRes)
class SelectSpies(Rule):
    def init(self):
        self.accomplishedPlayers=[]
    def applies(self, state, action, data, asignments):
        if action==Actions.SELECT and data.vote and state.lostTurns>=1:
            for p in data.team:
                pSpy=self.spyProbability(asignments, p)
                if pSpy>0.7:
                    return True
            return False
        else:
            return False
                
    def upd(self, asignments, state, action, data):
        pSpy=self.spyProbability(asignments, data.player)
        if pSpy>0 and pSpy<1:
            pRes=1-pSpy
            pVotoEnContraLuegoSaboteoSpia=self.spyStats.probability(Probabilities.VOTE_ON_TEAMS_WITH_SPIES,data.player,1.0,50)
            pVotoEnContraLuegoSaboteoRs=self.rsStats.probability(Probabilities.VOTE_ON_TEAMS_WITH_SPIES,data.player,0.2,50)
            denominator=(pSpy*(pVotoEnContraLuegoSaboteoSpia-pVotoEnContraLuegoSaboteoRs)+pVotoEnContraLuegoSaboteoRs)
            if denominator>0:
                newPSpy=(pVotoEnContraLuegoSaboteoSpia*pSpy)/denominator
                newPRs=1-newPSpy
                for a in asignments:
                    if data.player in a.spies:
                        a.probability=a.probability*(newPSpy/pSpy)
                    else:
                        a.probability=a.probability*(newPRs/pRes)


    
 
class VoteData:
    def __init__(self,player,vote):
        self.player=player
        self.vote=vote                                                          
class SelectData:
    def __init__(self,player,team):
        self.player=player
        self.team=team

class SelectedTeam:
    def __init__(self,leader,team):
        self.leader=leader
        self.team=team
class Magi(Bot):
    """This is the base class for your AI in THE RESISTANCE.  To get started:
         1) Derive this class from a new file that will contain your AI.  See
            bots.py for simple stock AI examples.

         2) Implement mandatory API functions below; you must re-implement
            those that raise exceptions (i.e. vote, select, sabotage).

         3) If you need any of the optional callback API functions, implement
            them (i.e. all functions named on*() are callbacks).

       Aside from parameters passed as arguments to the functions below, you 
       can also access the game state via the self.game variable, which contains
       a State class defined in game.py.

       For debugging, it's recommended you use the self.log variable, which
       contains a python logging object on which you can call .info() .debug()
       or warn() for instance.  The output is stored in a file in the #/logs/
       folder, named according to your bot. 
    """
    
    globalSpyPlayerStats=PlayerStats()
    globalResistancePlayerStats=PlayerStats()
   
    lastSelected=None
    sabotageIdx=0
    selectSabotageIdx=3
    noSabotearIdx=5
    noSelSabotearIdx=4
    definitiveSpies=[]
    baseModelsPerName={}
    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players  List of all players in the game including you.
        @param spies    List of players that are spies, or an empty list.
        """
        self.gatheringInfo=False
        self.spies=spies
        self.updSpyStats=self.globalSpyPlayerStats.copy()
        self.updResistanceStats=self.globalResistancePlayerStats.copy()
        self.gameState=GameState(players)
        self.deceives=[]
        self.trusts={}
        self.definitiveSpies=[]
        self.playerStats={}
        self.hmms=[]
        self.otherPlayers=self.others()
        a=self.otherPlayers[0]
        b=self.otherPlayers[1]
        c=self.otherPlayers[2]
        d=self.otherPlayers[3]
        abCDModel=PlayerAsignment([a,b],[c,d])
        acBDModel=PlayerAsignment([a,c],[b,d])
        adBCModel=PlayerAsignment([a,d],[b,c])
        bcadModel=PlayerAsignment([b,c],[a,d])
        bdacModel=PlayerAsignment([b,d],[a,c])
        cdabModel=PlayerAsignment([c,d],[a,b])
        self.players=players
        self.hmms.append(abCDModel)
        self.hmms.append(acBDModel)
        self.hmms.append(adBCModel)
        self.hmms.append(bcadModel)
        self.hmms.append(bdacModel)
        self.hmms.append(cdabModel)
        self.changed= True
        self.gameState.lostTurns=0
        self.bestModel=None
        self.rules=[]
        spyStats=self.globalSpyPlayerStats
        rsStats=self.globalResistancePlayerStats
        
        self.rules.append(SabotageRule(spyStats,rsStats))
        self.rules.append(VoteAgainsMissionLeader(spyStats,rsStats))
        self.rules.append(AcomplishedRule(spyStats,rsStats))
        self.rules.append(LastTryVote(spyStats,rsStats))
        self.rules.append(NoVoteOnSuccessfullTeamTeams(spyStats,rsStats))
        self.rules.append(VoteOnSpies(spyStats,rsStats))
        self.rules.append(SelectSpies(spyStats,rsStats))
        self.rules.append(NoVoteOnTeamsOfThreeWhereHeIsNot(spyStats,rsStats))
        self.burned=False
        self.otherBurned=False
        self.otherSelected=False
        pass
    def bestPrediction(self):
        if self.changed:
            bestModels=[]
            bestIdx=1000
            for md in self.hmms:
                tstIdx=0.0
                for md2 in self.hmms:
                    if md2!=md:
                        for p in md2.spies:
                            if p not in md.spies:
                                tstIdx+=md2.probability*self.updSpyStats.probability(Probabilities.SABOTAGE,p,1.0,30)
                                
                if tstIdx<bestIdx:
                    bestModels=[md]
                    bestIdx=tstIdx
                elif tstIdx==bestIdx:
                    bestModels.append(md)
            self.bestModel=bestModels[0]  
            self.changed=False
            #if bestIdx<=0.1:
                #assert False
                            
        return self.bestModel
    def bestPredictionB(self):
        if self.changed:
            self.bestModel=None
            bestIdx=-1
            for md in self.hmms:
                tstIdx=md.probability
                if tstIdx>bestIdx:
                    self.bestModel=md
                    bestIdx=tstIdx
            self.changed=False
            #if bestIdx<=0.1:
                #assert False
                            
        return self.bestModel
    def onMissionAttempt(self, mission, tries, leader):
        """Callback function when a new turn begins, before the
        players are selected.
        @param mission  Integer representing the mission number (1..5).
        @param tries    Integer count for its number of tries (1..5).
        @param leader   A Player representing who's in charge.
        """
        self.gameState.tries=tries
        self.gameState.turn=mission
        self.gameState.leader=leader
        pass

    def select(self, players, count):
        """Pick a sub-group of players to go on the next mission.
        @param players  The list of all players in the game to pick from.
        @param count    The number of players you must now select.
        @return list    The players selected for the upcoming mission.
        """
        others=[p for p in self.others() if p not in self.spies]
        
        if self.spy:                    
            if self.burned:
                return [self]+ random.sample(others,count-1)
            elif self.otherBurned and not self.otherSelected:
                return [p for p in self.spies if p != self]+ random.sample(others,count-1)
            elif not self.otherBurned and not self.otherSelected:
                return [p for p in self.spies if p != self]+ random.sample(others,count-1)
            else:
                return random.sample(self.spies,1)+ random.sample(others,count-1)
        
            
        otherResistanceMembers=self.predictedResistance()
        #random.shuffle(otherResistanceMembers)
        assert otherResistanceMembers[1] is not None,"error en la ressitencia"
       
        if count==3:                    
            return [self,otherResistanceMembers[0],otherResistanceMembers[1]]
        else:
            return [self,otherResistanceMembers[0]]

    def onTeamSelected(self, leader, team):
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        @param leader   The leader in charge for this mission.
        @param team     The team that was selected by the current leader.
        """
        self.lastLeader=leader
        self.lastSelected=team
        self.gameState.team=team
        self.gameState.leader=leader
        self.updResistanceStats.update(Probabilities.SELECT_HIMSELF,leader,leader in team,self.gameState)
        self.gameState.selectionResults.append(SelectedTeam(leader,team))
        if leader!=self and leader in self.spies:
            self.otherSelected=True
        pass
    def predictedSpies(self):
        return self.bestPrediction().spies
    def predictedResistance(self):
        return self.bestPrediction().resistance
    def vote(self, team):
        """Given a selected team, decide whether the mission should proceed.
        @param team      List of players with index and name. 
        @return bool     Answer Yes/No.
        """
       
        if self.game.tries == 5:
            return True
        if self==self.game.leader:
            return True
        if self.spy and len(team) == 3 and len([p for p in team if p in self.spies])>0:
            return True
        elif self.spy and len(team)==3:
            return False
        
        if len(team) == 3 and not self.index in [p.index for p in team]:
            return False
        
        if self.spy and not self in team:
            return random.random()>0.5
        
        if self.gameState.lostTurns>0:
            for p in team:
                if p in self.predictedSpies():
                    return False            
                
        return True    
    def onVoteComplete(self, votes):
        """Callback once the whole team has voted.
        @param votes        Boolean votes for each player (ordered).
        """
        self.gameState.voteComplete(votes)
        
        for i,p in enumerate(self.players):
            vote=votes[i]
            if p not in self.gameState.team:
                self.updSpyStats.update(Probabilities.VOTE_ON_THREETEAMS_WHEREHE_ISNT,p,vote,self.gameState)
            for r in self.rules:
                r.updateModel(self.hmms,self.gameState,Actions.VOTE,VoteData(p,vote))
        
        self.changed=True
        
        pass

    def sabotage(self):
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        @return bool        Yes to shoot down a mission.
        """
        self.burned=True
        return self.spy and not len([p for p in self.gameState.team if p in self.spies]) > 1

    def onMissionComplete(self, sabotaged):
        """Callback once the players have been chosen.
        @param selected     List of players that participated in the mission.
        @param sabotaged    Integer how many times the mission was sabotaged.
        """
        
        for p in self.gameState.team:
            self.updSpyStats.update(Probabilities.SABOTAGE,p,sabotaged>0,self.gameState)
        
        self.gameState.missionedTeams.append(MissionedTeam(self.gameState.team,sabotaged))
        self.gameState.turn=self.game.turn
        for r in self.rules:
            if sabotaged>0:
                r.updateModel(self.hmms,self.gameState,Actions.SABOTAGE,sabotaged)
            else:
                r.updateModel(self.hmms,self.gameState,Actions.ACCOMPLISHED,sabotaged)
        self.changed=True
        if sabotaged>0:
            self.gameState.lostTurns+=1
        
        if self in self.gameState.team:
            self.burned=True
        for p in self.spies:
            if p!=self and p in self.gameState.team:
                self.otherBurned=True
        
        pass

    def onGameComplete(self, win, spies):
        """Callback once the game is complete, and everything is revealed.
        @param win          Boolean if the Resistance won.
        @param spies        List of only the spies in the game.
        """
        
        for i,votes in enumerate(self.gameState.votes):
            votedTeam=self.gameState.selectedTeams[i]
            spiesInTeam=len([s for s in votedTeam if s in spies])
            
            for j,vote in enumerate(votes):
                player=self.gameState.players[j]
                spyInTeam=player in spies
                
                if spiesInTeam==0:
                    self.updSpyStats.update(Probabilities.VOTE_ON_RESISTENCE,player,vote,None)
                    self.updResistanceStats.update(Probabilities.VOTE_ON_RESISTENCE,player,vote,None)
                elif spiesInTeam>=1:
                    if spyInTeam:
                        self.updSpyStats.update(Probabilities.VOTE_ON_HE,player,vote,None)
                        self.updResistanceStats.update(Probabilities.VOTE_ON_HE,player,vote,None)
                    else:
                        self.updSpyStats.update(Probabilities.VOTE_ON_OTHER_SPIES,player,vote,None)
                        self.updResistanceStats.update(Probabilities.VOTE_ON_OTHER_SPIES,player,None)
                    self.updSpyStats.update(Probabilities.VOTE_ON_SPIES,player,vote,self.gameState)
                    self.updResistanceStats.update(Probabilities.VOTE_ON_SPIES,player,vote,None)    
                    if spiesInTeam>1:
                        self.updSpyStats.update(Probabilities.VOTE_ON_TWOSPY_TEAM,player,vote,None)
                        self.updResistanceStats.update(Probabilities.VOTE_ON_TWOSPY_TEAM,player,vote,None)
        for ms in self.gameState.missionedTeams:
            #ms=MissionedTeam()
            spiesOnTeam=len([s for s in ms.team if s in spies])
            if spiesOnTeam>1:
                for p in ms.team:
                    self.updSpyStats.update(Probabilities.SABOTAGE_ON_TWO_SPIE_TEAM,p,spiesOnTeam<ms.sabotages,None)
                
        for sel in self.gameState.selectionResults:
            #sel=SelectedTeam()
            spiesInTeam=len([s for s in sel.team if s in spies])
            self.updSpyStats.update(Probabilities.SELECT_SPIES,sel.leader,spiesInTeam>0,self.gameState)
            self.updResistanceStats.update(Probabilities.SELECT_SPIES,sel.leader,spiesInTeam>0,self.gameState)
        for p in self.game.players:
            if p in spies and p.name in self.updSpyStats.playersStats:
                self.globalSpyPlayerStats.playersStats[p.name]=self.updSpyStats.playersStats[p.name]
            elif p.name in self.updResistanceStats.playersStats:
                self.globalResistancePlayerStats.playersStats[p.name]=self.updResistanceStats.playersStats[p.name]       
        
                
         
        pass


class InvalidatorOracle(object):

    def __init__(self, game, bot):
        self.game = game
        self.bot = bot

    def selection(self, config):
        """Rate teams chosen by the leader, assuming a particular configuration.
        Zero means the selection is not suspicious, and positive values indicate
        higher suspicion levels."""

        all_spies = self.bot.getSpies(config)
        team_spies = [s for s in self.game.team if s in all_spies]
        if self.game.leader in all_spies and len(team_spies) != 1:
            return 1.0, [(1.0, "%s, assuming a spy, did not pick a mission with spies.")] 
        return 0.0, []

    def voting(self, config, votes):
        """Assess the votes of a player, assuming a particular configuration.
        Zero means no suspicious activity and positive values indicate high
        suspicion levels."""
        score = 0.0
        factors = []

        all_spies = self.bot.getSpies(config)
        team_spies = [s for s in self.game.team if s in all_spies]
        for p, v in zip(self.game.players, votes):
            # This is a spy, who voted for a mission, that had no spies.
            if p in all_spies and v and not team_spies:
                score += 1.0
                factors.append((1.0, "%s, assuming a spy, voted for a mission that had no assumed spies." % (p.name)))
            # This is a spy, who did not vote a mission, that had spies.
            if p in all_spies and not v and team_spies:
                score += 1.0
                factors.append((1.0, "%s, assuming a spy, did not vote a mission that had assumed spies." % (p.name)))
            # This is a Resistance guy who did not vote up the fifth try.
            if self.game.tries == 5 and p not in all_spies and not v:
                score += 2.0
            # This is a Resistance guy who voted up a mission without him!
            if p not in all_spies and len(self.game.team) == 3 and \
               p not in self.game.team and v:
                score += 2.0
        return score, factors

    def sabotages(self, config, sabotaged):
        spies = [s for s in self.game.team if s in self.bot.getSpies(config)]
        score = max(0, sabotaged - len(spies)) * 100.0
        if score > 0.0:
            return score, [(score, "%s participated in a mission that had %i sabotages." % (self.game.team, sabotaged))]
        else:
            return 0.0, []


class InvalidatorAdviser(object):

    def __init__(self, game, bot):
        self.game = game
        self.bot = bot

    def vote(self, team):
        if self.bot.spy:
            spies = [s for s in team if s in self.bot.spies]
            if len(spies) > 0 and (self.game.losses == 2 or self.game.wins == 2):
                self.bot.log.debug("Taking a risk since the game could finish.")
                return True
            
            if self.game.tries == 5:
                self.bot.log.debug("Voting up the last mission because Resistance would.")
                return True

            if len(team) == 3:
                self.bot.log.debug("Voting strongly about this team because it's size 3!")
                return self.bot in team

        else: # not self.bot.spy

            # Always approve our own missions.
            if self.game.leader == self.bot:
                self.bot.log.debug("Approving my own mission selection.")
                return True

            # As resistance, always pass the fifth try.
            if self.game.tries == 5:
                self.bot.log.debug("Voting up the last mission to avoid failure.")
                return True
        
        return None


class Invalidator(Bot):

    def onGameRevealed(self, players, spies):
        self.oracle = InvalidatorOracle(self.game, self)
        self.adviser = InvalidatorAdviser(self.game, self)
        self.players = players
        self.spies = spies

        # Count the number of times each configuration was apparently invalidated.
        self.invalidations = {k: 0.0 for k in permutations([True, True, False, False])}
        # This is used to help justify decisions in hybrid human/bot matches.
        self.factors = {k: [] for k in permutations([True, True, False, False])}

    def likeliest(self, configurations):
        ranked = sorted(configurations, key = lambda c: self.invalidations[c])
        invalidations = self.invalidations[ranked[0]]
        return [r for r in ranked if self.invalidations[r] == invalidations]

    def select(self, players, count):
        likely = self.likeliest(self.invalidations.keys())
        self.log.debug("Selecting randomly from these Resistance teams:")
        for c in likely:
            self.log.debug("  %s = %0.2f (%i)" % (self.getResistance(c), self.invalidations[c], len(self.factors[c])))
        config = random.choice(likely)

        if self.factors[config]:
            self.log.debug("Chosen configuration had these factors:")
            for s, f in self.factors[config]:
                self.log.debug("%0.2f - %s" % (s, f))
        return [self] + random.sample(self.getResistance(config), count-1)

    def onTeamSelected(self, leader, team):
        for config in self.invalidations:
            score, factors = self.oracle.selection(config)
            self.invalidations[config] += score
            self.factors[config].extend(factors)

    def vote(self, team): 
        advice = self.adviser.vote(team)
        if advice:
            return advice

        # Count the scores of configurations where no spies are selected. 
        scores = []
        matches = []
        for config in self.invalidations:
            if len([s for s in team if s in self.getSpies(config)]) == 0:
                scores.append(self.invalidations[config])
                matches.append(config)
        if not scores:
            self.log.debug("No configuration matches this selection!")
            return False

        # Establish whether this meets the criteria for selection...
        score = min(scores)
        threshold = min(self.invalidations.values())
        if score <= threshold:
            self.log.debug("This selection scores %s under threshold %f." % (scores, threshold))
            return True
        else:
            self.log.debug("This selection scores %s above threshold %0.2f." % (scores, threshold))
            for config in matches:
                self.log.debug("Possible configuration for %s:" % (str(self.getResistance(config))))
                for s, f in self.factors[config]:
                    self.log.debug("  %0.2f - %s" % (s, f))
            self.log.debug("Options for Resistance were:\n%s" % ("\n".join(["  %s = %0.2f (%i)" % (str(self.getResistance(c)), t, len(self.factors[c])) for c, t in self.invalidations.items() if t == threshold])))
            return False

    def onVoteComplete(self, votes):
        for config in self.invalidations:
            score, factors = self.oracle.voting(config, votes)
            self.invalidations[config] += score
            self.factors[config].extend(factors)

    def getSpies(self, config):
        return set([player for player, spy in zip(self.others(), config) if spy])

    def getResistance(self, config):
        return set([player for player, spy in zip(self.others(), config) if not spy])

    def onMissionComplete(self, sabotaged):
        for config in self.invalidations:
            score, factors = self.oracle.sabotages(config, sabotaged)
            self.invalidations[config] += score
            self.factors[config].extend(factors)

    def sabotage(self):
        # If there's a chance of losing or winning, don't slow-play!
        if self.game.wins == 2 or self.game.losses == 2:
            self.log.debug("There's a chance of winning or losing.")
            return True
        if len(self.game.team) == 2 and self.game.turn == 1:
            self.log.debug("Missions of size two are too risky...")
            return False
        spies = [s for s in self.game.team if s in self.spies]
        if len(spies) > 1:
            self.log.debug("Too many spies, can't coordinate!")
            return False
        self.log.debug("All other checks failed, why not since I'm a spy?")
        return True


class HartBot(Bot):
    threshold = 0.1
    increaseValue = 0.35
    decreaseValue = 0.25
    initialSuspection = 0.5
        
    '''
    This bot was created by Jonas Hartmann for the RESISTANCE competition
    realized by the AiGameDev.com for the GameAiConference in Vienna 2012
    '''
    def onGameRevealed(self, players, spies):
        self._logInfo("Game was revealed for me -> " + str(self))
        
        # initialize variables
        self.team = None
        self.suspection = self._createSuspectionDict(players, spies)
        self._logSuspection()
        self.players = players
        # save only the other spies        
        self.spies = [p for p in spies if p.index not in [self.index]]
        
        if self.spy:
            self._logInfo("Other spies: ")
            for player in self.spies:
                self._logInfo(player)
       
        self._logInfo("Other players in this game: ")
        for player in self.others():
            self._logInfo(player)

        pass
    
    def select(self, players, count):
        """Pick a sub-group of players to go on the next mission.
        @param players  The list of all players in the game to pick from.
        @param count    The number of players you must now select.
        @return list    The players selected for the upcoming mission.
        """
        # As a spy, pick myself and others who are not spies.
        if self.spy:
            self._logInfo("[TEAM SELECTION] Me and resistance team...")
            others = [p for p in self.others() if p not in self.spies]
            return [self] + random.sample(others, count - 1)
            
        # As a resistance, I will choose a team which I could trust
        self._logInfo("[TEAM SELECTION] selecting the most trusted...")
        return self._selectMostTrustableTeam(count)
        
    def vote(self, team):
        """Given a selected team, decide whether the mission should proceed.
        @param team      List of players with index and name. 
        @return bool     Answer Yes/No.
        """
        # Always approve our own missions.
        if self.game.leader == self:
            self._logInfo ("Approving our own mission...")
            return True

        # Both types of factions have constant behavior on the last try.
        if self.game.tries == 5:
            self._logInfo ("This mission must pass or not pass! Returning " + str(not self.spy))
            return not self.spy
        
        # As a spy, only approve if there is a spy in the mission  
        if self.spy:      
            self._logInfo ("Approving mission only if there is a spy")
            if self in team:
                return True
            return len([sp for sp in team if sp in self.spies]) > 0

        # If I'm not on the team and it's a team of 3, there is a spy in the team!
        if len(team) == 3 and not self.index in [p.index for p in team]:
            self._logInfo("Not approving mission")
            return False        
        
        # Do not vote for suspicious teams
        for p in team:
            if self.suspection[p.index] > self.threshold:
                self._logInfo("Not approving mission, he is probably a spy -> " + str(p))
                return False
        
        self._logInfo("Approving mission")
        return True
    
    def sabotage(self):
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        @return bool        Yes to shoot down a mission.
        """
        # Always take the chance to sabotage the mission
        self._logInfo("Sabotage!!")
        return self.spy

    def onMissionComplete(self, sabotaged):
        self._adjustSuspection(self.team, sabotaged)
        self._logSuspection()
        
        self._logInfo("Mission completed")
        if sabotaged:
            self._logInfo("Mission was sabotaged " + str(sabotaged) + " times")
        else:
            self._logInfo("Mission succeeded")
        
        self._logInfo("Mission team: " + str(self.team))
        self._logInfo("-----------------------------")
        
        pass
    
    
    def onMissionAttempt(self, mission, tries, leader):
        """Callback function when a new turn begins, before the
        players are selected.
        @param mission  Integer representing the mission number (1..5).
        @param tries    Integer count for its number of tries (1..5).
        @param leader   A Player representing who's in charge.
        """
        pass
    
    def onTeamSelected(self, leader, team):
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        @param leader   The leader in charge for this mission.
        @param team     The team that was selected by the current leader.
        """
        self.team = team
        self._logInfo("Selected mission team: " + str(self.team))
        
        return
    
    def onVoteComplete(self, votes):
        """Callback once the whole team has voted.
        @param votes        Boolean votes for each player (ordered).
        """
        pass
    
    def onGameComplete(self, win, spies):
        """Callback once the game is complete, and everything is revealed.
        @param win          Boolean if the Resistance won.
        @param spies        List of only the spies in the game.
        """
        if win:
            self._logInfo("GAME COMPLETE! RESISTANCE WON!")
        else:
            self._logInfo("GAME COMPLETE! SPIES WON!")
        
        self.team = None
        self.suspection = None
        pass
    
    ''' =================== '''
    '''  Private functions  '''
    ''' =================== '''
    
    def _selectMostTrustableTeam(self, count):
        """ Select the players which have the lowest suspicious value
        @param count The number of players you must select
        @return list The players selected
        """
        sortedSuspection = sorted(self.suspection.items(), key=lambda t : t[1])
        sortedIndexes = [k for k, v in sortedSuspection]
        return [p for p in self.players if p.index in sortedIndexes[0:count]]
    
    def _createSuspectionDict(self, players, spies):
        """
        @return: dictionary
        """
        if self.spy:
            # As a spy I already know everything
            sdict = {}
            for player in players:
                if player in spies:
                    sdict[player.index] = 1.0
                else:
                    sdict[player.index] = 0.0
            return sdict
            
        # As a resistance...
        sdict = {}
        for p in players:
            if p.index == self.index:
                sdict[p.index] = 0.0
            else:
                sdict[p.index] = self.initialSuspection
        return sdict
        
    def _increaseSuspection(self, player, teamSize):
        # ignore myself
        if self.index == player.index:
            return
        
        index = player.index
        self._logInfo("Increasing suspection of player: " + str(player))
        self.suspection[index] = self.suspection[index] + self.increaseValue
            
    def _decreaseSuspection(self, player):
        # ignore myself
        if self.index == player.index:
            return
        
        index = player.index
        self._logInfo("Decreasing suspection of player: " + str(player))
        self.suspection[index] = self.suspection[index] - self.decreaseValue
        if (self.suspection[index] < 0):
            self.suspection[index] = 0.0

    def _adjustSuspection(self, team, sabotaged):
        """ Will be called after each mission to adjust the way I 
        think about the other players
        """
        if self.spy:
            # As a spy I already know everything, so I just move on
            return
        
        
        # As a resistance...
        
        # If the mission was not sabotaged, trust a little more in the team
        if not sabotaged:
            self._logInfo("(THOUGHT) They seem to be nice guys...")
            for p in team:
                self._decreaseSuspection(p)
            return
       
        # If everyone in the team sabotaged the mission, then they are for sure spies
        if sabotaged == len(team):
            self._logInfo("(THOUGHT) They are all spies!!!")
            for p in team:
                self.suspection[p.index] = 10.0
            return
            
        # The mission was sabotaged and there were only 2 in the mission and I was one of them
        if len(team) == 2 and (self.index in [p.index for p in team]):
            self._logInfo("(THOUGHT) The other is a spy!")
            other = [p for p in team if p.index != self.index]
            self.suspection[other[0].index] = 10.0
            return
        
        # The mission was sabotaged and we were part of a big team, so don't trust the others
        if len(team) > 2 and (self.index in [p.index for p in team]):
            self._logInfo("(THOUGHT) One of the others is a spy!")
            others = [p for p in team if p.index not in [self.index]]
            for p in others:
                self._increaseSuspection(p, len(team) - 1)
            return
        
        # The mission was sabotaged but we don't exactly who, so don't trust the whole team
        self._logInfo("(THOUGHT) Someone is a spy...")
        for p in team:
            self._increaseSuspection(p, len(team))
        
        return

    def _logSuspection(self):
        self._logInfo("Suspection: " + str(self.suspection))
        pass
    
    def _logInfo(self, message):
        self.log.info(str(self) + ": " + str(message))
        pass



class BotPlan():

    def __init__(self):
        pass

    def init(self, game):
        self.game = game

    def load(self, rawPlan):

        # votes
        self.index = rawPlan[0]
        self.role = rawPlan[1]
        
        self.voteString = rawPlan[2:27]
        self.sabotageString = rawPlan[27:52]
        self.selectString = rawPlan[52:]

    def getVoteAction(self, missionId, attemptId):
        #print 'missionId:' + str(missionId)
        index = (missionId-1)*5 + (attemptId-1)
        #print 'index:' + str(index)
        actionId = self.voteString[index]
        return actionId == 1

    def getSabotageAction(self, missionId, attemptId):
        index = (missionId-1)*5 + (attemptId-1)
        actionId = self.sabotageString[index]
        return actionId == 1

    def getSelectAction(self, missionId, attemptId):
        index = (missionId-1)*5 + (attemptId-1)*5
        selection = self.selectString[index:index+5]

        playerIndex = 0
        players = []
        for n in selection:
            if n == 1:
                players = players + game.players[playerIndex]
            playerIndex = playerIndex + 1

        # print "selection:" + str(players)
        return players

class GrumpyBot(Bot):

    def __init__(self, game, index, spy):
        """Constructor called before a game starts.  It's recommended you don't
        override this function and instead use onGameRevealed() to perform
        setup for your AI.
        @param name     The public name of your bot.
        @param index    Your own index in the player list.
        @param spy      Are you supposed to play as a spy?
        """
        Player.__init__(self, self.__class__.__name__, index)
        self.game = game
        self.spy = spy

        self.spyplans = [  
                        "110000000000100000000000000100000000000000000000000000000000000000000000000000000000000000000000000000100010000000000000000000000000000000000000000000000000000000000000000000000"
        ]

        self.resplans = [
                        "100000010000000000000000000000000000000000000000000000000000000000000000000001101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"

        ]

        self.rawPlan = self.getPlansFor(index, spy)

        self.plan = BotPlan()
        self.plan.init(game)
        self.plan.load(self.rawPlan)

        self.log = logging.getLogger(str(self))
        if not self.log.handlers:
            try:
                output = logging.FileHandler(filename='logs/'+str(self)+'.xml')
                self.log.addHandler(output)
                self.log.setLevel(logging.DEBUG)
            except IOError:
                pass

    def getPlansFor(self, index, spy):

        if (spy):
            plans = self.spyplans
        else:
            plans = self.resplans

        candidates = [plan for plan in plans if plan.startswith(str(index))]
        
        if len(candidates) == 0:
            #print "random plan chosen"
            return random.choice(plans);
        else:
            #print "appropriate plan chosen"
            return random.choice(candidates)
    
    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players  List of all players in the game including you.
        @param spies    List of players that are spies, or an empty list.
        """
        
        role = "spy" if self.spy else "res"

        self.log.info("<Bot id=\"%d\" role=\"%s\">" %(self.index, role))
        # self.log.info("[%s] onGameRevealed()" %self)

    def onMissionAttempt(self, mission, tries, leader):
        """Callback function when a new turn begins, before the
        players are selected.
        @param mission  Integer representing the mission number (1..5).
        @param tries    Integer count for its number of tries (1..5).
        @param leader   A Player representing who's in charge.
        """
        #self.log.info("[%s] onMissionAttempt(), turn:%d, tries:%d, mission:%d, leader:%s" %(self, self.game.turn, self.game.tries, mission, leader))
       # self.log.info("<mission id=\"%d\" attempt=\"%d\" leader=\"%s\">" %(mission, tries, leader))

    def onMissionComplete(self, sabotaged):
        """Callback once the players have been chosen.
        @param selected     List of players that participated in the mission.
        @param sabotaged    Integer how many times the mission was sabotaged.
        """
        # self.log.info("[%s] onMissionComplete(), mission%d, tries:%d, sabotaged:%s" %(self, self.game.turn, self.game.tries, sabotaged))
        #self.log.info("</mission>")

    def select(self, players, count):
        """Pick a sub-group of players to go on the next mission.
        @param players  The list of all players in the game to pick from.
        @param count    The number of players you must now select.
        @return list    The players selected for the upcoming mission.
        """
        # self.log.info("[%s] select(), players:%s, count:%d" %(self, players, count))
        #others = random.sample(self.others(), count-1)
        #action = [self] + others
        #actionStr = [str(self.index) + "-" + self.name] + [p for p in others]
       
        action = self.plan.getSelectAction(self.game.turn, self.game.tries)
        if len(action) == 0:
            others = random.sample(self.others(), count-1)
            action = [self] + others
        
        #self.log.info("<select missionId=\"%d\" attemptId=\"%d\" count=\"%d\">%s</select>" %(self.game.turn, self.game.tries, count, actionStr))
        
        return action

    def onTeamSelected(self, leader, team):
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        @param leader   The leader in charge for this mission.
        @param team     The team that was selected by the current leader.
        """
        pass

    def vote(self, team): 
        """Given a selected team, decide whether the mission should proceed.
        @param team      List of players with index and name. 
        @return bool     Answer Yes/No.
        """ 
        # self.log.info("[%s] vote(), turn:%d, tries:%d, team:%s, res:%s" %(self, self.game.turn, self.game.tries, team, bool(self == self.game.leader)))
        # action = bool(self == self.game.leader)
        action = self.plan.getVoteAction(self.game.turn, self.game.tries)

        self.log.info("<vote missionId=\"%d\" attemptId=\"%d\" team=\"%s\">%s</vote>" %(self.game.turn, self.game.tries, team, action))
        return action

    def onVoteComplete(self, votes):
        """Callback once the whole team has voted.
        @param votes        Boolean votes for each player (ordered).
        """
        # self.log.info("[%s] onVoteComplete, turn:%d, tries:%d, votes:%s" %(self, self.game.turn, self.game.tries, votes))

    def sabotage(self):
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        @return bool        Yes to shoot down a mission.
        """
        #self.log.info("[%s] sabotage(), turn:%d, tries:%d, resWins:%d, spyWins:%d, team:%s" %(self, self.game.turn, self.game.tries, self.game.wins, self.game.losses, self.game.team ))
        #action = random.choice([True, False])
        action = self.plan.getSabotageAction(self.game.turn, self.game.tries)
        self.log.info("<sabotage missionId=\"%d\" attemptId=\"%d\">%s</sabotage>" %(self.game.turn, self.game.tries, action))
        return False 

    def onGameComplete(self, win, spies):
        """Callback once the game is complete, and everything is revealed.
        @param win          Boolean if the Resistance won.
        @param spies        List of only the spies in the game.
        """
        # self.log.info("[%s] onGameComplete(), turn:%d, tries:%d, win:%s, spies:%s" %(self, self.game.turn, self.game.tries, win, spies))
        winner = "res" if win else "spy"
        self.log.info("<winner>%s</winner>" %winner)
        self.log.info("</Bot>\n")

class Bot5Players(Bot):

    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players  List of all players in the game including you.
        @param spies    List of players that are spies, or an empty list.
        """
        self.memory = Memory(self, players, spies)
        self.entries = TeamEntries(players)
        if not self.spy:
            self.initialTrust = 1000
            self.entries.addTrust(self, self.initialTrust)
        self.log.info("Building behaviors")
        self.behavior = Bot5PlayersBehavior(self.game, self)



    def onMissionAttempt(self, mission, tries, leader):
        """Callback function when a new turn begins, before the
        players are selected.
        @param mission  Integer representing the mission number (1..5).
        @param tries    Integer count for its number of tries (1..5).
        @param leader   A Player representing who's in charge.
        """
        self.memory.currentMission = mission
        self.currentLeader = leader
        self.behavior.process(self.game, self, GamePhase.onMissionAttempt)

    def select(self, players, count):
        """Pick a sub-group of players to go on the next mission.
        @param players  The list of all players in the game to pick from.
        @param count    The number of players you must now select.
        @return list    The players selected for the upcoming mission.
        """
        self.memory.selectionCount = count
        return self.behavior.process(self.game, self, GamePhase.select)

    def onTeamSelected(self, leader, team):
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        @param leader   The leader in charge for this mission.
        @param team     The team that was selected by the current leader.
        """
        self.memory.currentTeam = list(team)
        self.memory.currentLeader = leader
        self.behavior.process(self.game, self, GamePhase.onTeamSelected)

    def vote(self, team):
        """Given a selected team, decide whether the mission should proceed.
        @param team      List of players with index and name.
        @return bool     Answer Yes/No.
        """
        self.memory.currentTeam = list(team)
        return self.behavior.process(self.game, self, GamePhase.vote)

    def onVoteComplete(self, votes):
        """Callback once the whole team has voted.
        @param votes        Boolean votes for each player (ordered).
        """
        self.memory.votes = votes
        self.behavior.process(self.game, self, GamePhase.onVoteComplete)

    def sabotage(self):
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        @return bool        Yes to shoot down a mission.
        """
        return self.behavior.process(self.game, self, GamePhase.sabotage)


    def onMissionComplete(self, sabotaged):
        """Callback once the players have been chosen.
        @param selected     List of players that participated in the mission.
        @param sabotaged    Integer how many times the mission was sabotaged.
        """
        self.memory.lastSabotage = sabotaged
        self.behavior.process(self.game, self, GamePhase.onMissionComplete)

    def onGameComplete(self, win, spies):
        """Callback once the game is complete, and everything is revealed.
        @param win          Boolean if the Resistance won.
        @param spies        List of only the spies in the game.
        """
        #have we found some patterns?
        self.behavior.process(self.game, self, GamePhase.onGameComplete)
#
#
#
class TeamEntry:
    def __init__(self, player):
        self.player = player
        self.count = 0


class TeamEntries:
    """Histogram filter"""
    def __init__(self, players):
        self.entries = dict()
        for p in players:
            self.entries[p] = 0

    def addTrust(self, player, value):
        self.entries[player] += value

class RuleStatistics:
    """Just like team entries, but used for get the most used rules"""
    def __init__(self):
        self.entries = dict()
        self.total = 0

    def ruleFired(self, ruleFired):
        if self.entries.has_key(ruleFired):
            self.entries[ruleFired] += 1
        else:
            self.entries[ruleFired] = 1
        self.total += 1

    def __repr__(self):
        result = "TOTAL RULES FIRED %i\n" % (self.total)
        for key, value in sorted(self.entries.iteritems(), key=lambda (k,v): (v,k)):
            result += "RULE: %s  TIMES:  %s\n" % (key, value)
        return result

rulesStatistics = RuleStatistics()

#
#===========
# BEHAVIORS
#===========
#

class ResistanceBaseBehavior:
    """The base class for all behaviors"""
    def __init__(self, game, owner, priority = 0):
        self.owner = owner
        self.game = game
        self.priority = priority

    def process(self, game, owner, phase):
        return (False, None)

    def __cmp__(self, other):
        assert isinstance(other, ResistanceBaseBehavior)
        if self.priority < other.priority:
            return -1
        elif self.priority < other.priority:
            return 1
        else:
            return 0

class ResistanceDelegateBehaviour(ResistanceBaseBehavior):
    """A behavior whose process function delegates the calculation to the function given"""
    def __init__(self, game, owner, priority = 0, delegateFunction=None):
        ResistanceBaseBehavior.__init__(self, game, owner, priority)
        self.delegateFunction = delegateFunction

    def process(self, game, owner, phase):
        if self.delegateFunction:
            return self.delegateFunction(game, owner, phase)
        return (False, None)

class ResistanceCompositeBaseBehavior(ResistanceBaseBehavior):
    """Base class for behaviors that are composed for other more simple behaviors"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceBaseBehavior.__init__(self, game, owner, priority)
        self.children = children
        self.children.sort()


    def process(self, game, owner, phase):
        for behaviour in self.children:
            output = behaviour.process(game,owner,phase)
            if output[0]:
                rulesStatistics.ruleFired(behaviour.__class__.__name__)
                return output

        return (False, None)

class TrueBehavior(ResistanceBaseBehavior):
    def process(self, game, owner, phase):
        return (True, True)

class FalseBehavior(ResistanceBaseBehavior):
    def process(self, game, owner, phase):
        return (True, False)

#
# SELECTION BEHAVIORS
#
class RandomSelectionBehaviour(ResistanceBaseBehavior):
    """Just selects a team randomly"""
    def process(self, game, owner, phase):
        owner.log.info("A completely random selection.")
        return (True, random.sample(game.players, owner.memory.selectionCount))

class MeAndRandomSelectionBehaviour(ResistanceBaseBehavior):
    """Just selects me and the others randomly"""
    def process(self, game, owner, phase):
        return (True, [owner] + random.sample(owner.others(),  owner.memory.selectionCount - 1))

class OneSpyRandomSelectionBehavior(ResistanceBaseBehavior):
    """Selects one of the two spies randomly"""
    def process(self, game, owner, phase):
        others = owner.memory.others - owner.memory.spies
        return (True, random.sample(owner.memory.spies,1) + random.sample(others,  owner.memory.selectionCount - 1))

class OneSpySelectionBehavior(ResistanceBaseBehavior):
    """Selects one of the two spies randomly"""
    def process(self, game, owner, phase):
        spies = list(owner.memory.spies)
        sorted_by_trust = []
        for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
            if not(key in spies):
                sorted_by_trust.append((key, value))
        sorted_by_trust.reverse()

        less_suspicious = spies[0]
        if owner.entries.entries[less_suspicious] < owner.entries.entries[spies[0]]:
            less_suspicious = spies[1]
        team = [less_suspicious]

        for i in range(owner.memory.selectionCount - 1):
            team.append(sorted_by_trust[i][0])
        return (True, team)

class OneSpyLessSuspiciousSelectionBehavior(ResistanceBaseBehavior):
    """Selects the spy less suspicious or not played yet"""
    def process(self, game, owner, phase):
        if game.turns > 1 and game.wins == 0:
            spies = list(owner.memory.spies)
            sorted_by_trust = []
            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
                if not(key in spies):
                    sorted_by_trust.append((key, value))
            sorted_by_trust.reverse()

            index1 = spies[0].index
            index2 = spies[1].index
            plays1 = 0
            plays2 = 0
            for i in range(game.turn-1):
                if (i+1) == 2 or (i+1) == 4 or (i+1) == 5:
                    plays1 += 0.5 * owner.memory.selections.value[i][index1]
                    plays2 += 0.5 * owner.memory.selections.value[i][index2]
                else:
                    plays1 += owner.memory.selections.value[i][index1]
                    plays2 += owner.memory.selections.value[i][index2]

            less_suspicious = spies[0]
            if plays2 < plays1:
                less_suspicious = spies[1]
            team = [less_suspicious]

            for i in range(owner.memory.selectionCount - 1):
                team.append(sorted_by_trust[i][0])
            return (True, team)
        else:
            return(False, None)

class SpySelectionBehavior(ResistanceCompositeBaseBehavior):
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [OneSpySelectionBehavior(game, owner, 1),
                        OneSpyLessSuspiciousSelectionBehavior(game, owner, 0)]


class MeAndOnlyResitanceRandomSelectionBehaviour(ResistanceBaseBehavior):
    """Just selects me and the others resistance members randomly"""
    def process(self, game, owner, phase):
        if len(owner.memory.resistance) >= owner.memory.selectionCount - 1:
            return (True, [owner] + random.sample(owner.memory.resistance,  owner.memory.selectionCount - 1))
        else:
            return(False, None)

class OnlyResitanceRandomSelectionBehaviour(ResistanceBaseBehavior):
    """resistance members randomly"""
    def process(self, game, owner, phase):
        team = owner.memory.resistance | set([owner])
        if len(team) >= owner.memory.selectionCount:
            return (True, random.sample(team,  owner.memory.selectionCount))
        else:
            return(False, None)

class MeAndLessSuspiciousSelectionBehaviour(ResistanceBaseBehavior):
    """We haven't got enough info, me, resistance and others that might be not spies"""
    def process(self, game, owner, phase):
        others = owner.memory.others - owner.memory.resistance - owner.memory.spies
        if len(others) + len(owner.memory.resistance) >= owner.memory.selectionCount-1:
            return (True, [owner] + list(owner.memory.resistance) + random.sample(others,  owner.memory.selectionCount - 1 - len(owner.memory.resistance)))
        else:
            return(False, None)

class ResistanceSelectionWorstCase(ResistanceBaseBehavior):
    """Just selects me and the others randomly"""
    def process(self, game, owner, phase):
        others = owner.memory.others - owner.memory.resistance - owner.memory.spies
        return (True, [owner] + list(owner.memory.resistance) + list(others) + random.sample(owner.memory.spies, owner.memory.selectionCount - 1 - len(owner.memory.resistance) - len(others)))

class LessSuspiciousSelectionBehaviour(ResistanceBaseBehavior):
    """We haven't got enough info, me, resistance and others that might be not spies"""
    def process(self, game, owner, phase):
        sorted_by_trust = []
        for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
            if key != owner:
                sorted_by_trust.append((key, value))
        sorted_by_trust.reverse()
        #Do I trust on the second one?
        if sorted_by_trust[1][1] > 0 and game.losses < Game.NUM_LOSSES - 1:
            return (True, random.sample([owner, sorted_by_trust[0][0], sorted_by_trust[1][0]], owner.memory.selectionCount))
        else:
            #ok, me first then follow my feelings
            team = [owner]
            for i in range(owner.memory.selectionCount-1):
                team.append(sorted_by_trust[i][0])
            return (True, team)

class NotSelectedSelectionBehaviour(ResistanceBaseBehavior):
    """We cant afford lose, me with one of the group of three"""
    def process(self, game, owner, phase):
        #use this method if we haven't won any round
        if game.turn == 3 and game.wins == 0:
            sorted_by_trust = []
            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
                if key != owner:
                    sorted_by_trust.append((key, value))
            sorted_by_trust.reverse()
            if sorted_by_trust[0][1] <= 0.5:
                #I have no clue, anyone who hasn't been selected yet
                player = - 1
                trust = -999999
                for j in range(5):
                    for i in range(game.turn-1):
                        if owner.memory.selections.value[i][j] > 0 or j==owner.index:
                            break
                    if i == game.turn-1  and  owner.entries[game.players[j]] > trust:
                        player = j
                        trust = owner.entries[game.players[j]]

                if player >= 0:
                    return(True, [owner, game.players[player]])
                else:
                    #if all players have participated already, I'll select the most trusted one
                    return (False, None)
            else:
                #we have enough info to select the best ones
                return (False, None)
        else:
            return (False, None)

##class NotSelectedSelectionBehaviour(ResistanceBaseBehavior):
##    """We cant afford lose, me with one of the group of three"""
##    def process(self, game, owner, phase):
##        #use this method if we haven't won any round
##        if game.turn == 3 and game.wins == 0:
##            sorted_by_trust = []
##            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
##                if key != owner:
##                    sorted_by_trust.append((key, value))
##            sorted_by_trust.reverse()
##            if sorted_by_trust[0][1] <= 0.5:
##                #I have no clue, anyone who hasn't been selected yet
##                player = - 1
##                trust = -999999
##                for j in range(5):
##                    if owner.memory.selections.value[1][j] == 1 and j != owner.index-1 and owner.entries[game.players[j]] > trust:
##                        player = j
##                        trust = owner.entries[game.players[j]]
##
##                if player >= 0:
##                    return(True, [owner, game.players[player]])
##                else:
##                    #if all players have participated already, I'll select the most trusted one
##                    return (False, None)
##            else:
##                #we have enough info to select the best ones
##                return (False, None)
##        else:
##            return (False, None)

class MostUntrustedSelectionBehaviour(ResistanceBaseBehavior):
    """let's put the most suspicious together with the members that haven't played"""
    def process(self, game, owner, phase):
        #use this method if we haven't won any round
        if game.turn == 2 and game.wins == 0:
            sorted_by_trust = []
            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
                if key != owner:
                    sorted_by_trust.append((key, value))
            sorted_by_trust.reverse()
            #we could have lost but te result was two sabotages
            if sorted_by_trust[0][1] <= 0.5:
                #I have no clue, most suspicious together with others  who hasn't played
                all_players = set(game.players)
                players = set([])
                for j in range(5):
                    for i in range(game.turn-1):
                        if owner.memory.selections.value[i][j] > 0 or j==owner.index:
                            break
                    if i == game.turn-1:
                        players.add(game.players[j])

                diff = all_players - players
                if owner in diff:
                    diff.remove(owner)
                    team = random.sample(diff,  1) + random.sample(players,  2)
                else:
                    players.remove(owner)
                    team = random.sample(diff,  1) + random.sample(players,  2)
                return (True, team)
            else:
                #we have enough info to select the best ones
                return (False, None)
        else:
            return (False, None)


class ResistanceMemberSelectionBehavior(ResistanceCompositeBaseBehavior):
    """High-level behavior for resistance members"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [LessSuspiciousSelectionBehaviour(game, owner, 1),
                        MostUntrustedSelectionBehaviour(game, owner, 0),
                        #NotSelectedSelectionBehaviour(game, owner, 0)
                        ]

#
# VOTING BEHAVIORS
#
class RandomVotingBehaviour(ResistanceBaseBehavior):
    """Just returns true or false randomly"""
    def process(self, game, owner, phase):
        return (True, random.choice([True, False]))


class VotingBCResistanceMemberBehaviour(ResistanceBaseBehavior):
    """
    If it's the last try accept the team.
    Resistance members will accept the team, and spies could be easily spotted
    if the reject it
    """
    def process(self, game, owner, phase):
        # Base case, accept the mission anyway
        # a spy should be easily spotted if  it rejects the team
        # We are supposing resistance players have somekind of rationality
        if game.tries == Game.MAX_TURNS:
            return (True, True)
        else:
            return (False, None)


class JustOneSpyBehavior(ResistanceBaseBehavior):
    """Better avoid to have a team consisting only in spies"""
    def process(self, game, owner, phase):
        if len([p for p in owner.memory.currentTeam if p in owner.memory.spies]) == len(owner.memory.currentTeam):
            return (True, random.choice([True, False]))
        else:
            return (False, None)

class AtLeastOneSpyVotingBehavior(ResistanceBaseBehavior):
    def process(self, game, owner, phase):
        return (True, len([p for p in owner.memory.currentTeam if p in owner.memory.spies]) > 0)

class SpyVotingBCBehavior(ResistanceBaseBehavior):
    """I don't care with the number of spies (at least one) if it's our last mission"""
    def process(self, game, owner, phase):
        if game.losses == Game.NUM_LOSSES -1:
            return (True, len([p for p in owner.memory.currentTeam if p in owner.memory.spies]) > 0)
        else:
            return (False, None)

class SpyVotingBC2Behavior(ResistanceBaseBehavior):
    """First mission, first round and no spies-> coin flip"""
    def process(self, game, owner, phase):
        if game.tries == 1 and game.turn == 1 and len([p for p in owner.memory.currentTeam if p in owner.memory.spies]) == 0:
            return (True, random.choice([True, False]))
        else:
            return (False, None)

class SpyVotingBehavior(ResistanceCompositeBaseBehavior):
    """High-level behavior for voting as a spy"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [AtLeastOneSpyVotingBehavior(game, owner, 5),
                        JustOneSpyBehavior(game, owner, 4),
                        SpyVotingBCBehavior(game, owner, 0),
                        SpyVotingBC2Behavior(game, owner, 2),
                    VotingBCResistanceMemberBehaviour(game, owner, 1)]
        self.children.sort()

class ResistanceMemberBasicVoting(ResistanceBaseBehavior):
    """
    Just make sure the team doesn't have any bot who might be a spy
    (from my point of view)
    """
    def process(self, game, owner, phase):

        #how much i trust you
        leader_trust = owner.entries.entries[owner.memory.currentLeader]

        sorted_by_trust = []

        for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
            if key != owner:
                sorted_by_trust.append((key, value))

        if leader_trust < 0 and (( sorted_by_trust[0][0] ==  owner.memory.currentLeader ) or ( sorted_by_trust[1][0] ==  owner.memory.currentLeader )):
            #I don't trust you at all
            return (True, False )

        sorted_by_trust.reverse()

        #best fit (without me)
        max_trust = sorted_by_trust[0][1] + sorted_by_trust[1][1]

        trust = 0
        for member in owner.memory.currentTeam:
            if member != owner:
                trust += owner.entries.entries[member]


##      return (True, trust >= max_trust)
##        return (True, trust >= max_trust or trust >= 0)

        best_fit = set([owner, sorted_by_trust[0][0], sorted_by_trust[1][0]])

        #FIX. the magic number 2.5 is just because the maximum penalization for a mission failed is 5
        tolerance = 2.5
        return (True, set(owner.memory.currentTeam).issubset(best_fit) or abs(max_trust - trust) < tolerance)

class NotSelectedOnVotingBehaviour(ResistanceBaseBehavior):
    """We haven't won any round and not any clue, approve "new" pairs"""
    def process(self, game, owner, phase):
        #use this method if we haven't won any round
        if game.turn == 3 and game.wins == 0:
            sorted_by_trust = []
            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
                if key != owner:
                    sorted_by_trust.append((key, value))
            #sorted_by_trust.reverse()
            #if sorted_by_trust[0][1] <= 0.5:
            if owner.entries.entries[owner.memory.currentLeader] >= -3.5:
                v1 = owner.memory.selections.take([0,1,2,3,4],[owner.memory.currentTeam[0].index])
                v2 = owner.memory.selections.take([0,1,2,3,4],[owner.memory.currentTeam[1].index])
                #let's try this team if it has never selected before
                res = v1.transpose() * v2
                return (True, res.value[0][0] == 0)
##                if res.value[0][0] == 0:
##                    return (True, True)
##                else:
##                    return (False, None)
            else:
                #we have enough info to select the best ones
                return (False, None)
        else:
            return (False, None)

##class NotSelectedOnVotingBehaviour(ResistanceBaseBehavior):
##    """We haven't won any round and not any clue, approve "new" pairs"""
##    def process(self, game, owner, phase):
##        #use this method if we haven't won any round
##        if game.turn == 3 and game.wins == 0:
##            sorted_by_trust = []
##
##            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
##                if key != owner:
##                    sorted_by_trust.append((key, value))
##
##            sorted_by_trust.reverse()
##
##            best_fit = set([owner, sorted_by_trust[0][0], sorted_by_trust[1][0]])
##
##            return (True, set(owner.memory.currentTeam).issubset(best_fit))
##        else:
##            return(False, None)

class NotSelected3OnVotingBehaviour(ResistanceBaseBehavior):
    """I can't afford to lose, me with someone of the previous team"""
    def process(self, game, owner, phase):
        #use this method if we haven't won any round
        if game.turn == 3 and game.wins == 0:
            sorted_by_trust = []
            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
                if key != owner:
                    sorted_by_trust.append((key, value))
            sorted_by_trust.reverse()
            if sorted_by_trust[0][1] <= 0.5:
                players = set([])
                for j in range(5):
                    if owner.memory.selections.value[1][j] == 1 and owner.memory.selections.value[0][j] == 0  and j != owner.index:
                        players.add(game.players[j])
                    if (owner in owner.memory.currentTeam) and len(set(owner.memory.currentTeam).intersection(players))==1:
                    #or len(set(owner.memory.currentTeam).intersection(players))==2:
                        return(True, True)
                    else:
                        return(True, False)
            else:
                #we have enough info to select the best ones
                return (False, None)
        else:
            return (False, None)


class NotSelected2OnVotingBehaviour(ResistanceBaseBehavior):
    """second round, approve the most suspicious together"""
    def process(self, game, owner, phase):
        #use this method if we haven't won any round
        if game.turn == 2 and game.wins == 0:
            sorted_by_trust = []
            for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
                if key != owner:
                    sorted_by_trust.append((key, value))
            sorted_by_trust.reverse()
            if sorted_by_trust[0][1] <= 0.5:
                #I have no clue, most suspicious together with others  who hasn't played
                diff = set([])
                players = set([])
                for j in range(5):
                    if owner.memory.selections.value[0][j] == 0:
                        players.add(game.players[j])
                    else:
                        diff.add(game.players[j])

                inteam = owner in owner.memory.currentTeam
                indiff = owner in diff
                if inteam and indiff:
                    return (True, len(players.intersection(owner.memory.currentTeam)) == 2)
                elif inteam and not indiff:
                    players.remove(owner)
                    return (True, len(diff.intersection(owner.memory.currentTeam)) == 1)
                else:
                    return (True, len(diff.intersection(owner.memory.currentTeam)) == 1 and len(players.intersection(owner.memory.currentTeam)) == 2)
            else:
                #we have enough info to select the best ones
                return (False, None)
        else:
            return (False, None)

class VotingBCBehavior(ResistanceBaseBehavior):
    """Approve if I'm the leader"""
    def process(self, game, owner, phase):
        if owner == owner.memory.currentLeader:
            return (True, True)
        else:
            return (False, None)

class ResistanceMemberBCTeamMembers(ResistanceBaseBehavior):
    """
    If the team size is equal to the number of resistance members and I'm not
    part of team reject it
    """
    def process(self, game, owner, phase):
        sorted_by_trust = []
        for key, value in sorted(owner.entries.entries.iteritems(), key=lambda (k,v): (v,k)):
            if key != owner:
                sorted_by_trust.append((key, value))
        sorted_by_trust.reverse()
        if sorted_by_trust[0][1] > 0.5 or game.wins > 0:
            if len(owner.memory.currentTeam) == 3 and  (not (owner in owner.memory.currentTeam)):
                return (True, False)
            else:
                return (False, None)
        else:
            return (False, None)

##class ResistanceMemberBCTeamMembers(ResistanceBaseBehavior):
##    """
##    If the team size is equal to the number of resistance members and I'm not
##    part of team reject it
##    """
##    def process(self, game, owner, phase):
##        if len(owner.memory.currentTeam) == 3 and  (not (owner in owner.memory.currentTeam)):
##            return (True, False)
##        else:
##            return (False, None)

class ResistanceMemberVotingBehavior(ResistanceCompositeBaseBehavior):
    """High-level behavior for voting as a resistance member"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [TrueBehavior(game, owner, 10),
                        ResistanceMemberBasicVoting(game, owner, 4),
                        NotSelectedOnVotingBehaviour(game, owner, 3),
                        #NotSelected3OnVotingBehaviour(game, owner, 2),
                        NotSelected2OnVotingBehaviour(game, owner, 2),
                        ResistanceMemberBCTeamMembers(game, owner, 1),
                        VotingBCBehavior(game, owner, 1),
                        VotingBCResistanceMemberBehaviour(game, owner, 0)]
        self.children.sort()

#
# ONVOTECOMPLETED BEHAVIORS
#

class ResistanceMemberOnVotingBaseCase(ResistanceBaseBehavior):
    """
    Just a basic reasoning. If it was the last votation attempt,
    all the ones who have rejected the team could possibly be spies!
    """
    def process(self, game, owner, phase):
        # base case: see if any bot has rejected the team, they must be spies
        if game.tries == Game.MAX_TURNS and game.losses < Game.NUM_LOSSES:
            for i in range(len(owner.memory.votes)):
                if not owner.memory.votes[i]:
                     owner.entries.addTrust(game.players[i], -1)
        #the phase after the voting is mostly for reasoning, so we don't need stop other types of them
        return (False, None)


class ResistanceMemberOnVotingFirstTime(ResistanceBaseBehavior):
    """
    Why reject a team in the first place? Suspicious...
    """
    def process(self, game, owner, phase):
        # do you know something i dont?
        rejecteds = owner.memory.votes.count(False)
        if game.tries == 1 and game.turn == 1 and  rejecteds > 0 and rejecteds < 4:
            for i in range(len(owner.memory.votes)):
                if not owner.memory.votes[i]:
                     owner.entries.addTrust(game.players[i], -1)
        #the phase after the voting is mostly for reasoning, so we don't need stop other types of them
        return (False, None)

class ResistanceMemberOnVotingBehavior(ResistanceCompositeBaseBehavior):
    """Highest-level behavior for on vote completed"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [ResistanceMemberOnVotingBaseCase(game, owner, 0),
                         ResistanceMemberOnVotingFirstTime(game, owner, 0)]

#
# SABOTAGING BEHAVIORS
#
class ResistanceMemberSabotagingBehaviour(ResistanceBaseBehavior):
    """Just like FalseBehavior, and it's here just to have one behavior per game phase"""
    def process(self, game, owner, phase):
        #Resistance memebers never sabotage a mission
        return (True, False)

class SpySabotagingBehavior(ResistanceCompositeBaseBehavior):
    """Highest-level behavior for sabotaging"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [TrueBehavior(game, owner, 3),
                        TwoSpiesSabotagingBehavior(game, owner, 2),
                        OnlyTwoSpiesSabotagingBehavior(game, owner, 1),
                        SpySabotageBaseCaseBehavior(game, owner, 0)]
##        self.children = [TrueBehavior(game, owner, 0)]
        self.children.sort()

class SpySabotageBaseCaseBehavior(ResistanceBaseBehavior):
    """Spy base case for sabotaging. Sabotage the mission if it leads to an immediat victory"""
    def process(self, game, owner, phase):
        if game.losses == Game.NUM_LOSSES-1:
            return (True,True)
        else:
            return(False, None)

class OnlyTwoSpiesSabotagingBehavior(ResistanceBaseBehavior):
    """The team is composed only by spies!"""
    def process(self, game, owner, phase):
        if len([p for p in game.team if p in owner.memory.spies]) == len(game.team)  and game.wins < Game.NUM_WINS-1:
            #people won't trust me if I sabotage the mission
            return (True, False)
        else:
            return (False,None)

class TwoSpiesSabotagingBehavior(ResistanceBaseBehavior):
    """Two spies in a team of three. How suspicious am I?"""
    def process(self, game, owner, phase):
        otherSpy = list(owner.memory.spies - set([owner]))
        if len([p for p in game.team if p in owner.memory.spies]) == 2 and game.wins < Game.NUM_WINS-1:

            index1 = owner.index
            index2 = otherSpy[0].index
            plays1 = 0
            plays2 = 0
            for i in range(game.turn-1):
                if (i+1) == 2 or (i+1) == 4 or (i+1) == 5:
                    plays1 += 0.5 * owner.memory.selections.value[i][index1]
                    plays2 += 0.5 * owner.memory.selections.value[i][index2]
                else:
                    plays1 += owner.memory.selections.value[i][index1]
                    plays2 += owner.memory.selections.value[i][index2]

            #only sabotage if i'm more suspicious
            return (True, (plays1 > plays2) or (plays1==plays2 and owner.entries.entries[owner] < owner.entries.entries[otherSpy[0]]) or (random.choice([True, False])))
        else:
            return (False,None)

#
# ONMISSIONCOMPLETED BEHAVIORS
#

class OnMissionCompletedResistanceFailBCBehavior(ResistanceBaseBehavior):
    """Two members, two sabotages->two spies the most easy one"""
    def process(self, game, owner, phase):
        if owner.memory.lastSabotage == len(owner.memory.currentTeam):
            others = owner.memory.others - set(owner.memory.currentTeam)
            for member in owner.memory.currentTeam:
                owner.entries.addTrust(member, -5)
                #get the selections
                owner.memory.selections.value[game.turn - 1][member.index] = 1
            for member in others:
                owner.entries.addTrust(member, 5)
            return(True, None)
        return (False, None)


class OnMissionCompletedResistanceFailBC2Behavior(ResistanceBaseBehavior):
    """Me with spies!"""
    def process(self, game, owner, phase):
        if owner.memory.lastSabotage == (len(owner.memory.currentTeam)-1) and (owner in owner.memory.currentTeam):
            for member in owner.memory.currentTeam:
                owner.entries.addTrust(member, -5)
                #get the selections
                owner.memory.selections.value[game.turn - 1][member.index] = 1
            owner.entries.addTrust(owner.memory.currentLeader, -0.5)
            return(True, None)
        return (False, None)

class OnMissionCompletedResistanceFailBehavior(ResistanceBaseBehavior):
    """One spy in the team"""
    def process(self, game, owner, phase):
        if owner.memory.lastSabotage > 0:
            im_in_the_team = (owner in owner.memory.currentTeam)
            #im_in_the_team = 0
            for member in owner.memory.currentTeam:
                owner.entries.addTrust(member, -2/len(owner.memory.currentTeam) - im_in_the_team/3 )
                #get the selections
                owner.memory.selections.value[game.turn - 1][member.index] = 1
            owner.entries.addTrust(owner.memory.currentLeader, -0.5)

            for i in range(len(owner.memory.votes)):
                if owner.memory.votes[i]:
                    owner.entries.addTrust(game.players[i], -0.5)
                else:
                    owner.entries.addTrust(game.players[i], 0.5)
            return (True, None)
        return (False, None)

class OnMissionCompletedResistanceFail2Behavior(ResistanceBaseBehavior):
    """Three members (not me), two sabotages->two spies"""
    def process(self, game, owner, phase):
        if owner.memory.lastSabotage == 2:
            others = owner.memory.others - set(owner.memory.currentTeam)
            for member in owner.memory.currentTeam:
                #penalize a little bit more
                owner.entries.addTrust(member, -2)
                #get the selections
                owner.memory.selections.value[game.turn - 1][member.index] = 1
            for member in others:
                owner.entries.addTrust(member, 5)
            return(True, None)
        return (False, None)

class OnMissionCompletedResistanceBCBehavior(ResistanceBaseBehavior):
    """We won with this team, infer something!"""
    def process(self, game, owner, phase):
        if owner.memory.lastSabotage == 0:

            for member in owner.memory.currentTeam:
                owner.entries.addTrust(member, 4)
                #get the selections
                owner.memory.selections.value[game.turn - 1][member.index] = 1

            if not (owner.memory.currentLeader in owner.memory.currentTeam) and owner.memory.currentLeader!= owner:
                owner.entries.addTrust(owner.memory.currentLeader, 0.5)

            if len(owner.memory.currentTeam) == 3:
                others = owner.memory.others - set(owner.memory.currentTeam)
                for member in others:
                    owner.entries.addTrust(member, -4)

            for i in range(len(owner.memory.votes)):
                if owner.memory.votes[i]:
                    owner.entries.addTrust(game.players[i], 0.5)
                else:
                    owner.entries.addTrust(game.players[i], -0.5)
            return (True, None)
        return (False, None)

class OnMissionCompletedResistanceBehavior(ResistanceCompositeBaseBehavior):
    """High-level"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [OnMissionCompletedResistanceFailBehavior(game, owner, 5),
                        OnMissionCompletedResistanceFail2Behavior(game, owner, 1),
                        OnMissionCompletedResistanceFailBC2Behavior(game, owner, 0),
                        OnMissionCompletedResistanceFailBCBehavior(game, owner, 0),
                        OnMissionCompletedResistanceBCBehavior(game, owner, 0),]
        self.children.sort()








class OnMissionCompletedSpyBehavior(ResistanceCompositeBaseBehavior):
    """High-level  behavior for spies"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        self.children = [OnMissionCompletedResistanceFailBehavior(game, owner, 1),
                        OnMissionCompletedResistanceFailBCBehavior(game, owner, 0),
                        OnMissionCompletedResistanceBCBehavior(game, owner, 0),]
        self.children.sort()


#
# HIGHEST-LEVEL ONE
#
class Bot5PlayersBehavior(ResistanceCompositeBaseBehavior):
    """The highest level behavior"""
    def __init__(self, game, owner, priority = 0, children=[]):
        ResistanceCompositeBaseBehavior.__init__(self, game, owner, priority,children)
        #init sub-behaviors depending on being a spy or not
        if self.owner.spy:
            self.children = [ResistanceBaseBehavior(game, owner, GamePhase.onGameRevealed),
                            ResistanceBaseBehavior(game, owner, GamePhase.onMissionAttempt),
                            #OneSpyRandomSelectionBehavior(game, owner, GamePhase.select),
                            #OneSpySelectionBehavior(game, owner, GamePhase.select),
                            SpySelectionBehavior(game, owner, GamePhase.select),
                            ResistanceBaseBehavior(game, owner, GamePhase.onTeamSelected),
                            SpyVotingBehavior(game, owner, GamePhase.vote),
                            ResistanceBaseBehavior(game, owner, GamePhase.onVoteComplete),
                            SpySabotagingBehavior(game, owner, GamePhase.sabotage),
                            #ResistanceBaseBehavior(game, owner, GamePhase.onGameComplete),
                            OnMissionCompletedSpyBehavior(game, owner, GamePhase.onMissionComplete),
                            ResistanceBaseBehavior(game, owner, GamePhase.onGameComplete)]
        else:
            self.children = [ResistanceBaseBehavior(game, owner, GamePhase.onGameRevealed),
                            ResistanceBaseBehavior(game, owner, GamePhase.onMissionAttempt),
                            ResistanceMemberSelectionBehavior(game, owner, GamePhase.select),
                            ResistanceBaseBehavior(game, owner, GamePhase.onTeamSelected),
                            ResistanceMemberVotingBehavior(game, owner, GamePhase.vote),
                            ResistanceMemberOnVotingBehavior(game, owner, GamePhase.onVoteComplete),
                            FalseBehavior(game, owner, GamePhase.sabotage),
                            OnMissionCompletedResistanceBehavior(game, owner, GamePhase.onMissionComplete),
                            ResistanceBaseBehavior(game, owner, GamePhase.onGameComplete)]

    def process(self, game, owner, phase):
        return self.children[phase].process(game,owner,phase)[1]



class GamePhase:
    PHASES = 9
    onGameRevealed=0
    onMissionAttempt=1
    select=2
    onTeamSelected=3
    vote=4
    onVoteComplete=5
    sabotage=6
    onMissionComplete=7
    onGameComplete=8

class Memory:
    """Or should I say BlackBoard system"""
    _currentLeader = None
    _currentTeam = None
    _currentMission = None
    _selectionCount = None
    _votes = None
    _lastSabotage = None

    def __init__(self, owner, players, spies):
        self.players = players
        self.spies = spies
        self.spiesIndex = [spy.index for spy in spies]
        self.others = set(owner.others())
        self.resistance = set()
        self.selections = matrix()
        self.selections.zero(5,5)

    @property
    def currentLeader(self):
        return self._currentLeader

    @currentLeader.setter
    def currentLeader(self, value):
        self._currentLeader = value

    @property
    def currentTeam(self):
        return self._currentTeam

    @currentTeam.setter
    def currentTeam(self, value):
        self._currentTeam = value

    @property
    def currentMission(self):
        return self._currentMission

    @currentMission.setter
    def currentMission(self, value):
        self._currentMission = value

    @property
    def selectionCount(self):
        return self._selectionCount

    @selectionCount.setter
    def selectionCount(self, value):
        self._selectionCount = value

    @property
    def votes(self):
        return self._votes

    @votes.setter
    def votes(self, value):
        self._votes = value

    @property
    def lastSabotage(self):
        return self._lastSabotage

    @lastSabotage.setter
    def lastSabotage(self, value):
        self._lastSabotage

class BehaviorsStatistics:
    """See the most-used behaviors"""
    def __init__(self):
        self.total = 0
        self.entries = dict()

#============
# UTILS
#============
#---------------------------------------
# Matrix class from Sebastrian Thrun's Udacity course (self-driving car)
#
class matrix:

    # implements basic operations of a matrix class

    # ------------
    #
    # initialization - can be called with an initial matrix
    #

    def __init__(self, value = [[]]):
        self.value = value
        self.dimx  = len(value)
        self.dimy  = len(value[0])
        if value == [[]]:
            self.dimx = 0

    # ------------
    #
    # makes matrix of a certain size and sets each element to zero
    #

    def zero(self, dimx, dimy):
        if dimy == 0:
            dimy = dimx
        # check if valid dimensions
        if dimx < 1 or dimy < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            self.dimx  = dimx
            self.dimy  = dimy
            self.value = [[0.0 for row in range(dimy)] for col in range(dimx)]

    # ------------
    #
    # makes matrix of a certain (square) size and turns matrix into identity matrix
    #

    def identity(self, dim):
        # check if valid dimension
        if dim < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            self.dimx  = dim
            self.dimy  = dim
            self.value = [[0.0 for row in range(dim)] for col in range(dim)]
            for i in range(dim):
                self.value[i][i] = 1.0
    # ------------
    #
    # prints out values of matrix
    #

    def show(self, txt = ''):
        for i in range(len(self.value)):
            print txt + '['+ ', '.join('%.3f'%x for x in self.value[i]) + ']'
        print ' '

    # ------------
    #
    # defines elmement-wise matrix addition. Both matrices must be of equal dimensions
    #

    def __add__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimx != other.dimx:
            raise ValueError, "Matrices must be of equal dimension to add"
        else:
            # add if correct dimensions
            res = matrix()
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] + other.value[i][j]
            return res

    # ------------
    #
    # defines elmement-wise matrix subtraction. Both matrices must be of equal dimensions
    #

    def __sub__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimx != other.dimx:
            raise ValueError, "Matrices must be of equal dimension to subtract"
        else:
            # subtract if correct dimensions
            res = matrix()
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] - other.value[i][j]
            return res

    # ------------
    #
    # defines multiplication. Both matrices must be of fitting dimensions
    #

    def __mul__(self, other):
        # check if correct dimensions
        if self.dimy != other.dimx:
            raise ValueError, "Matrices must be m*n and n*p to multiply"
        else:
            # multiply if correct dimensions
            res = matrix()
            res.zero(self.dimx, other.dimy)
            for i in range(self.dimx):
                for j in range(other.dimy):
                    for k in range(self.dimy):
                        res.value[i][j] += self.value[i][k] * other.value[k][j]
        return res


    # ------------
    #
    # returns a matrix transpose
    #

    def transpose(self):
        # compute transpose
        res = matrix()
        res.zero(self.dimy, self.dimx)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[j][i] = self.value[i][j]
        return res

    # ------------
    #
    # creates a new matrix from the existing matrix elements.
    #
    # Example:
    #       l = matrix([[ 1,  2,  3,  4,  5],
    #                   [ 6,  7,  8,  9, 10],
    #                   [11, 12, 13, 14, 15]])
    #
    #       l.take([0, 2], [0, 2, 3])
    #
    # results in:
    #
    #       [[1, 3, 4],
    #        [11, 13, 14]]
    #
    #
    # take is used to remove rows and columns from existing matrices
    # list1/list2 define a sequence of rows/columns that shall be taken
    # is no list2 is provided, then list2 is set to list1 (good for
    # symmetric matrices)
    #

    def take(self, list1, list2 = []):
        if list2 == []:
            list2 = list1
        if len(list1) > self.dimx or len(list2) > self.dimy:
            raise ValueError, "list invalid in take()"

        res = matrix()
        res.zero(len(list1), len(list2))
        for i in range(len(list1)):
            for j in range(len(list2)):
                res.value[i][j] = self.value[list1[i]][list2[j]]
        return res

    # ------------
    #
    # creates a new matrix from the existing matrix elements.
    #
    # Example:
    #       l = matrix([[1, 2, 3],
    #                  [4, 5, 6]])
    #
    #       l.expand(3, 5, [0, 2], [0, 2, 3])
    #
    # results in:
    #
    #       [[1, 0, 2, 3, 0],
    #        [0, 0, 0, 0, 0],
    #        [4, 0, 5, 6, 0]]
    #
    # expand is used to introduce new rows and columns into an existing matrix
    # list1/list2 are the new indexes of row/columns in which the matrix
    # elements are being mapped. Elements for rows and columns
    # that are not listed in list1/list2
    # will be initialized by 0.0.
    #

    def expand(self, dimx, dimy, list1, list2 = []):
        if list2 == []:
            list2 = list1
        if len(list1) > self.dimx or len(list2) > self.dimy:
            raise ValueError, "list invalid in expand()"

        res = matrix()
        res.zero(dimx, dimy)
        for i in range(len(list1)):
            for j in range(len(list2)):
                res.value[list1[i]][list2[j]] = self.value[i][j]
        return res

    # ------------
    #
    # Computes the upper triangular Cholesky factorization of
    # a positive definite matrix.
    # This code is based on http://adorio-research.org/wordpress/?p=4560
    #

    def Cholesky(self, ztol= 1.0e-5):

        res = matrix()
        res.zero(self.dimx, self.dimx)

        for i in range(self.dimx):
            S = sum([(res.value[k][i])**2 for k in range(i)])
            d = self.value[i][i] - S
            if abs(d) < ztol:
                res.value[i][i] = 0.0
            else:
                if d < 0.0:
                    raise ValueError, "Matrix not positive-definite"
                res.value[i][i] = sqrt(d)
            for j in range(i+1, self.dimx):
                S = sum([res.value[k][i] * res.value[k][j] for k in range(i)])
                if abs(S) < ztol:
                    S = 0.0
                res.value[i][j] = (self.value[i][j] - S)/res.value[i][i]
        return res

    # ------------
    #
    # Computes inverse of matrix given its Cholesky upper Triangular
    # decomposition of matrix.
    # This code is based on http://adorio-research.org/wordpress/?p=4560
    #

    def CholeskyInverse(self):

        res = matrix()
        res.zero(self.dimx, self.dimx)

        # Backward step for inverse.
        for j in reversed(range(self.dimx)):
            tjj = self.value[j][j]
            S = sum([self.value[j][k]*res.value[j][k] for k in range(j+1, self.dimx)])
            res.value[j][j] = 1.0/ tjj**2 - S/ tjj
            for i in reversed(range(j)):
                res.value[j][i] = res.value[i][j] = \
                    -sum([self.value[i][k]*res.value[k][j] for k in \
                              range(i+1,self.dimx)])/self.value[i][i]
        return res

    # ------------
    #
    # comutes and returns the inverse of a square matrix
    #
    def inverse(self):
        aux = self.Cholesky()
        res = aux.CholeskyInverse()
        return res

    # ------------
    #
    # prints matrix (needs work!)
    #
    def __repr__(self):
        return repr(self.value)


class GarboA(Bot):
    def onGameRevealed(self, players, spies):
        self.spies = spies
        self.players = players
        self.suspicion = {}
        self.missions_done = {}
        for player in players:
            self.suspicion[player]=0
            self.missions_done[player]=0
        self.me = [p for p in players if p.index == self.index][0]
        self.others = [p for p in players if not p.index == self.index]
        
        self.log2 = self.log
        def nolog(params):
            pass
        if self.spy:
            self.log.debug = nolog
        self.log.debug("---GAME %s---" % str(self.me))
        self.log.debug(spies)
    def select(self, players, count):
        if(self.spy):
            # spy with lowest suspicion + resistance with lowest suspicion
            return [sorted(list(self.spies),key=lambda i:self.suspicion[i])[0]] + sorted([i for i in players if i not in self.spies],key=lambda i:self.suspicion[i])[0:count-1]
        else:
            # if we have to pick 3 players, we should include ourselves to minimize having a spy
            # if we are among the least suspicious players, we include ourselves
            if count > 2 or (self.me in sorted(self.suspicion.items(),key=lambda i:i[1])[0:2]):
                # myself + the least supicious of the remaining players 
                return [self.me] + sorted(self.others,key=lambda i:self.suspicion[i])[0:count-1]
            else:
                # least suspicious players 
                return sorted(self.game.players,key=lambda i:self.suspicion[i])[0:count]

    def onMissionComplete(self, sabotaged):
        # fact spy = 1000
        # fact res= -1000
        # suspicion ~100
        team = self.game.team
        no_team = [p for p in self.game.players if p not in self.game.team]
        for p in self.game.team:
            self.missions_done[p] += 1
        if sabotaged:
            # the team is suspicious
            if self.me not in self.game.team:
                if len(team)==2 and sabotaged==2:            
                    for p in team:
                        self.suspicion[p] += 1000
                    for p in no_team:
                        self.suspicion[p] -= 1000
                if len(team)==2 and sabotaged==1:
                    for p in team:
                        self.suspicion[p] += 50
                if len(team)==3 and sabotaged==2:
                    for p in team:
                        self.suspicion[p] += 100
                    for p in no_team:
                        self.suspicion[p] -= 1000
                if len(team)==3 and sabotaged==1:
                    for p in team:
                        self.suspicion[p] += 25
            else:
                if len(team)==2 and sabotaged==1:
                    for p in [p for p in team if not p==self.me]:
                        self.suspicion[p] += 1000
                    self.suspicion[self.me] += 50
                if len(team)==3 and sabotaged==2:
                    for p in [p for p in team if not p==self.me]:
                        self.suspicion[p] += 1000
                    self.suspicion[self.me] += 100
                    for p in no_team:
                        self.suspicion[p] -= 1000
                if len(team)==3 and sabotaged==1:
                    for p in [p for p in team if not p==self.me]:
                        self.suspicion[p] += 50
                    self.suspicion[self.me] += 25
            # the leader and voters are suspicious
            self.voterPenalites(team)
        else:
            if len(team)==2:
                for p in team:
                    self.suspicion[p] -= 50
            else:
                for p in team:
                    self.suspicion[p] -= 25   
        self.log.debug("---ROUND %d %s ---" % (self.game.turn,str(self.me)))
        self.log.debug(sabotaged)
        self.log.debug(self.game.team)
        self.log.debug(self.suspicion)
    def voterPenalites(self,team):
            if self.game.turn>1 and max([self.suspicion[i] for i in team])>=100:
                self.suspicion[self.game.leader] += 40
                for p in filter(lambda i:self.last_votes[i.index] and not i==self.game.leader, self.game.players):
                    self.suspicion[p]+=20
            if len(team)==3 and max([self.suspicion[i] for i in team])>=50:
                for i in [i for i in self.players if i not in team and self.last_votes[i.index]]:
                    self.suspicion[i] += 700 
    def onGameComplete(self, win, spies):
        self.log.debug("---END %s ---" % str(self.me))
        self.log.debug(self)
        self.log.debug(spies)
        self.log.debug(self.suspicion) 
    def sabotage(self):
        return self.spy 
    def vote(self, team):
        self.log.debug("---VOTING %d %s ---" % (self.game.tries,str(self.me)))
        self.log.debug(self.game.leader)
        self.log.debug(team)
        if self.game.tries == 5: return True
        suspects = sorted(self.game.players,key=lambda i:self.suspicion[i])
        suspects2 = [p for p in suspects if not p==self.me]
        if self.spy:
            return len([p for p in team if p in self.spies])>0
        else:
            # We need data
            if self.game.turn == 1: return True 
            if suspects[-1] in team: return False
            if len(team)==2:
                if self.me in team and suspects[-2] in team: return False    
            elif len(team)==3:
                # There is going to be a spy if it is not all resistance members, we want in
                if self.me not in team:
                    return False
                elif suspects2[-1] in team or suspects[-2] in team:
                    return False
        return True  
    def onVoteComplete(self, votes):
        our = votes[self.index]
        if self.game.turn>1 and self.game.tries<5:
            for i in self.game.players:
                if not votes[i.index]==our:
                    self.suspicion[i]+=8
            team = self.game.team
            team_filtered = filter(lambda i:self.suspicion[i]<750,team)
            if self.me not in team and len(team)==3 and len(team_filtered)<3:              
                not_team_voted_true = [i for i in self.game.players if i not in team and votes[i.index]]
                for i in not_team_voted_true:
                    self.suspicion[i]+=500
            if self.me not in team and len(team)==3:
                for i in [i for i in self.game.players if i not in team and votes[i.index]]:
                    self.suspicion[i]+=400
            # When there there is a spy NOT in the team, voting AGAINST the team, let's assume there are no spies on the team
            if len(filter(lambda i:self.suspicion[i]>750 and i not in team and not votes[i.index],self.game.players))>0:
                for i in self.game.team:  
                    self.suspicion[i]-=600
            
        self.last_votes = votes
        self.log.debug("---VOTES %s ---" % str(self.me))
        self.log.debug(votes)
