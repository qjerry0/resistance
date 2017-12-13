import random
import itertools

from player import Bot


def permutations(config):
    """Returns unique elements from a list of permutations."""
    return list(set(itertools.permutations(config)))


class Simpleton(Bot):
    """A bot that does logical reasoning based on the known spies and the
    results from the mission sabotages."""

    def onGameRevealed(self, players, spies):
        self.spies = spies
        self.configurations = permutations([True, True, False, False])

    def getSpies(self, config):
        assert len(config) == 4
        assert all([type(c) is bool for c in config])
        return [player for player, spy in zip(self.others(), config) if spy]

    def getResistance(self, config):
        assert len(config) == 4
        assert all([type(c) is bool for c in config])
        return [player for player, spy in zip(self.others(), config) if not spy]

    def _validateSpies(self, config, team, sabotaged):
        spies = [s for s in team if s in self.getSpies(config)]
        return len(spies) >= sabotaged

    def _validateNoSpies(self, config, team):
        spies = [s for s in team if s in self.getSpies(config)]
        return len(spies) == 0

    def select(self, players, count):
        if self.configurations:
            # Pick one of the many options first, who knows...    
            config = self._select(self.configurations)
            # Now pick some random players with or without myself.
            return [self] + random.sample(self.getResistance(config), count - 1)
        else:
            # assert self.spy
            resistance = [p for p in self.others() if p not in self.spies]
            return [self] + random.sample(resistance, count - 1)

    def _select(self, configurations):
        """This is a hook for inserting more advanced reasoning on top of the
        maximal amount of logical reasoning you can perform."""
        return random.choice(configurations)
        
    def _acceptable(self, team):
        """Determine if this team is an acceptable one to vote for..."""
        current = [c for c in self.configurations if self._validateNoSpies(c, team)]
        return bool(len(current) > 0)

    def vote(self, team):
        if self.game.tries == 5:
            return not self.spy

        # If it's not acceptable, then we have to shoot it down.
        if not self._acceptable(team):
            # self.log.debug("%s: This configuration is not acceptable." % ("SPY" if self.spy else "RST"))
            return False
        # Otherwise we randomly pick a course of action, who knows?
        else:
            return self._vote(team)

    def _vote(self, team):
        """This is a hook for providing more complex voting once logical
        reasoning has been performed."""
        return True

    def onMissionComplete(self, sabotaged):
        before = len(self.configurations)
        self.configurations = [c for c in self.configurations if self._validateSpies(c, self.game.team, sabotaged)]
        after = len(self.configurations)
        # self.log.debug("%s: Filtered out %i configurations, %i left." % ("SPY" if self.spy else "RST", after - before, after))
        # self.log.debug("%r" % [self.getSpies(c) for c in self.configurations])

    def sabotage(self):
        return True


class Trickerton(Simpleton):
    """A configuration-based logical reasoning bot with better spy behavior."""

    # def _select(self, configuration):
    #   TODO: If spy, don't select configuration with only spies.

    def _vote(self, team):
        return self == self.game.leader or random.choice([True, False])

    def sabotage(self):
        return self.game.turn > 1


class Bounder(Simpleton):
    """Idea of upper and lower bounds shamelessly stolen from Peter Cowling. :-)
       This is an implementation of his bot for comparison and modeling."""

    def onGameRevealed(self, players, spies):
        self.spies = spies

        # The set of possible assignments around the table, for:
        #   - PESSIMISTIC: All teams except those 100% proven to be spies.
        self.pessimistic = permutations([True, True, False, False])
        #   - OPTIMISTIC: The teams we don't suspect to be spies without guarantees.
        self.optimistic = permutations([True, True, False, False])

    def select(self, players, count):
        if self.optimistic:
            config = random.choice(self.optimistic)
        else:
            assert len(self.pessimistic) > 0
            config = random.choice(self.pessimistic)
        return [self] + random.sample(self.getResistance(config), count-1)

    def _validate(self, config, team, sabotaged, optimistic):
        spies = [s for s in team if s in self.getSpies(config)]
        if optimistic:
            return len(spies) == sabotaged
        else:
            return len(spies) >= sabotaged

    def vote(self, team): 
        # Determine if this is an acceptable thing to vote for...
        def acceptable(configurations, optimistic):
            current = [c for c in configurations if self._validate(c, team, 0, optimistic)]
            return bool(len(current) > 0)

        # Try our best-case options first, otherwise fall back...
        if self.optimistic:
            return acceptable(self.optimistic, True)
        elif self.pessimistic:
            return acceptable(self.pessimistic, False)
        else:
            return random.choice([True, False])

    def onMissionComplete(self, sabotaged):
        if self.spy:
            return

        self.optimistic = [c for c in self.optimistic if self._validate(c, self.game.team, sabotaged, True)]
        self.pessimistic = [c for c in self.pessimistic if self._validate(c, self.game.team, sabotaged, False)]

    def sabotage(self):
        return True


class Logicalton(Bot):
    """Applies some simple logic rules such as not allowing teams that include a
    team who've failed a mission previously."""

    def onGameRevealed(self, players, spies):
        self.players = players
        self.spies = spies
        self.team = None
        self.taboo = []

    def select(self, players, count):
        # As a spy, pick myself and others who are not spies.
        if self.spy:
            others = [p for p in players if p not in self.spies]
            return [self] + random.sample(others, count-1)

        team = []
        # If there was a previously selected successful team, pick it! 
        if self.team: # and not self._discard(self.team):
            team = [p for p in self.team if p.index != self.index and p not in self.spies]
        # If the previous team did not include me, reduce it by one.
        if len(team) > count-1:
            team = self._sample([], team, count-1)
        # If there are not enough people still, pick another randomly.
        if len(team) == count-1:
            return [self] + team
        # Try to put together another team that combines past winners and not spies.
        others = [p for p in players if p != self and p not in (set(team) | self.spies)]
        return self._sample([self] + team, others, count-1-len(team))

    def _sample(self, selected, candidates, count):
        while True:
            selection = selected + random.sample(candidates, count)
            if self._discard(selection):
                continue
            return selection
        # The selected team has been discarded, meaning there's a problem with
        # the selected candidates.
        assert False, "Problem in team selection."
        
    def _discard(self, team):
        # Has a subset of the proposed team failed a mission before?
        for t in self.taboo:
            if set(t).issubset(set(team)):
                return True
        return False

    def vote(self, team): 
        # As a spy, vote for all missions that include one spy!
        if self.spy:
            return len([p for p in team if p in self.spies]) > 0
        # Always approve our own missions.
        if self.game.leader == self:
            return True
        # As resistance, always pass the fifth try.
        if self.game.tries == 5:
            return True
        # If there's a known spy on the team.
        if set(team).intersection(self.spies):
            return False
        # Taboo list of past suspicious teams.
        if self._discard(team):
            return False
        # If I'm not on the team and it's a team of 3!
        if len(team) == 3 and not self.index in [p.index for p in team]:
            return False
        # Otherwise, just approve the team and get more information. 
        return True

    def onVoteComplete(self, votes):
        self.team = None
    
    def onMissionComplete(self, sabotaged):
        if self.spy:
            return
        # Keep track of the team if it's successful
        if not sabotaged:
            self.team = self.game.team
            return
        
        # Divide the team into known spies and suspects
        suspects = [p for p in self.game.team if p not in self.spies and p != self]
        spies = [p for p in self.game.team if p in self.spies]

        if sabotaged >= len(suspects) + len(spies):
            # We have more thumbs down than suspects and spies!
            for spy in [s for s in suspects if s not in self.spies]:
                self.spies.add(spy)
        else:
            # Remember this specific failed teams so we can taboo search.
            self.taboo.append([p for p in self.game.team if p != self])

    def sabotage(self):
        return self.spy










class Jerry(Bot):

    def onGameRevealed(self, players, spies):
        self.spies = spies
        self.players = players
        self.configurations = {k: 0.0 for k in permutations([True, True, False, False])}

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
                unlikely += 2.0
            if leader not in spies and len(spiesOnMission) == len(team):
                unlikely += 1.0
            if leader not in spies and len(spiesOnMission) + 1 == len(team):
                unlikely += 0.5
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
                return True #may want to do case for tries == 4?!
        return None

    def vote(self, team): 
        if self.expertVote(team) is not None:
            return self.expertVote(team)

        scores = []
        for c in self.configurations:
            if len([s for s in team if s in self.getSpies(c)]) == 0:
                scores.append(self.configurations[c])
        if not scores:
            return False
        threshold = min(self.configurations.values())
        return min(scores) <= threshold

    def onVoteComplete(self, votes):
        for c in self.configurations:
            unlikely = 0.0
            spies = self.getSpies(c)
            spiesOnMission = self.getSpiesOnMission(c)
            for p, v in zip(self.game.players, votes):
                if p in spies and len(spiesOnMission) == 1 and not v:
                    unlikely += 0.7
                if p in spies and len(spiesOnMission) == 0 and v:
                    unlikely += 0.7
                if p in spies and len(spiesOnMission) > 1 and v:
                    unlikely += 0.3
                if p not in spies and self.game.tries == 5 and not v:
                    unlikely += 3.0
                if p not in spies and p not in self.game.team and v:
                    if len(self.game.team) <= 2:
                        unlikely += 2.0
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
                unlikely += 2.0
            if leader in spies and len(spiesOnMission) == 0:
                unlikely += 2.0
            self.configurations[c] += unlikely


    def getSpies(self, config):
        return set([player for player, spy in zip(self.others(), config) if spy])

    def getSpiesOnMission(self, config):
        spies = self.getSpies(config)
        return [s for s in spies if s in self.game.team]

# not needed 
    def getResistance(self, config):
        return [p for p, s in zip(self.others(), config) if not s]
