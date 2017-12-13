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