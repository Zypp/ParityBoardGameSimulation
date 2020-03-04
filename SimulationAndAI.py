import random, copy
import numpy as np
from multiprocessing import Process, Value, Array

verbose = False

#recursively calculate the best move for each player for the rest of the game and play them out
#use with ~4 turns remaining to avoid state space explosion
def perfectAI(state, usedPremove=True, first=True):
    hand = state.hands[state.activePlayer-1]
    #make a list of potential moves
    #each move is an array with card, station1, station2 and switch
    moves = potentialMoves(state)
    
    activePlayer = state.activePlayer-1
    otherPlayer = (activePlayer+1)%2     
          
    if (len(state.hands[0])+len(state.hands[1])) > 1: #not in a leaf
        scoresMatrix = [] 
        movesMatrix = [[] for _ in range(len(moves))]
        premoveMatrix = []
        for i in range(len(moves)):
            move = moves[i]
            stateCopy = copy.deepcopy(state)
            if move[4] == -1:
                stateCopy.premoveBonus[activePlayer] += 1
            stateCopy.makeMove(move[0],move[1],move[2],move[3])
            scores,movesResult = perfectAI(stateCopy,usedPremove=(move[4]>=0),first=False)
            scoresMatrix.append(scores)
            movesMatrix[i] = movesResult
            premoveBonus = list(stateCopy.premoveBonus)
            premoveMatrix.append(premoveBonus)
        
        bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)
        
        #if we are capturing a station but the AI last turn did not, retroactively decide they defended that station
        if moves[bestMove][4] >= 0 and not usedPremove:
            del moves[bestMove]
            del premoveMatrix[bestMove]
            del scoresMatrix[bestMove]
            bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)
                
        if moves[bestMove][4] == -1:
            state.premoveBonus[activePlayer] += 1
            
        state.premoveBonus[activePlayer] += premoveMatrix[bestMove][activePlayer]
        state.premoveBonus[otherPlayer] += premoveMatrix[bestMove][otherPlayer]
            
        movesArray = movesMatrix[bestMove]
        move = moves[bestMove]
        movesArray.append(move)

        if not first:
            return scoresMatrix[bestMove],movesArray
        else:
            while movesArray: 
                
                move = movesArray.pop()
                if verbose:
                    state.printState()
                    print('perfect move',move)
                #capturing station first
                if move[4] >= 0:
                    state.makePreMove(move[4])
                state.makeMove(move[0],move[1],move[2],move[3])
    else: #we are in a leaf
        scoresMatrix = [] 
        premoveMatrix = []
        for move in moves:
            stateCopy = copy.deepcopy(state)
            stateCopy.makeMove(move[0],move[1],move[2],move[3])
            c = getCycle(stateCopy.switches, stateCopy.game)
            scores = stateCopy.cyclesPoints[c]
            scoresMatrix.append(scores)    
            premovesBonus = list(state.premoveBonus)
            if move[4] == -1:
                premovesBonus[activePlayer] += 1
            premoveMatrix.append(premovesBonus)
        bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)
        #if we are capturing a station but the AI last turn did not, retroactively decide they defended that station
        if moves[bestMove][4] >= 0 and not usedPremove:
            del moves[bestMove]
            del premoveMatrix[bestMove]
            del scoresMatrix[bestMove]
            bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)
                
        if moves[bestMove][4] == -1:
            state.premoveBonus[activePlayer] += 1
        return scoresMatrix[bestMove],[moves[bestMove]]

#make the best move, planning n turns ahead
def nTurnAI(state, first=False, n=2, usedPremove=True):
    assert n < (len(state.hands[0]) + len(state.hands[1]))
    
    activePlayer = state.activePlayer-1
    otherPlayer = (activePlayer+1)%2  
    
    moves = potentialMoves(state)
    
    if n == 0:
        scoresMatrix = [] 
        premoveMatrix = []
        for move in moves:
            stateCopy = copy.deepcopy(state)
            stateCopy.makeMove(move[0],move[1],move[2],move[3])
            c = getCycle(stateCopy.switches, stateCopy.game)
            scores = stateCopy.cyclesPoints[c]
            scoresMatrix.append(scores)   
            premovesBonus = list(stateCopy.premoveBonus)
            if move[4] == -1:
                premovesBonus[activePlayer] += 1
            premoveMatrix.append(premovesBonus)
        bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)
        if not usedPremove and moves[bestMove][4] >= 0:
            del moves[bestMove]
            del premoveMatrix[bestMove]
            del scoresMatrix[bestMove]
            bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)
        
        if first:
            move = moves[bestMove]
            state.makeMove(move[0],move[1],move[2],move[3])
        else:
            return scoresMatrix[bestMove]
    else:
        scoresMatrix = []
        premoveMatrix = []
        for i in range(len(moves)):
            move = moves[i]
            stateCopy = copy.deepcopy(state)
            stateCopy.makeMove(move[0],move[1],move[2],move[3])
            if move[4] == -1:
                stateCopy.premoveBonus[activePlayer] += 1
            scores = nTurnAI(stateCopy,n=n-1,usedPremove=(move[4]>=0))
            scoresMatrix.append(scores)
            premoveMatrix.append(stateCopy.premoveBonus)
        bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)

        if not usedPremove and moves[bestMove][4] >= 0:
            del moves[bestMove]
            del premoveMatrix[bestMove]
            del scoresMatrix[bestMove]
            bestMove = optimalMove(moves,scoresMatrix,activePlayer,otherPlayer,premoveMatrix)
            
        if not first:
            return scoresMatrix[bestMove]
        else:
            move = moves[bestMove]
            if move[4] == -1:
                state.premoveBonus[activePlayer] += 1
                state.usedPremove = True
            else:
                state.usedPremove = False
            if verbose:
                state.printState()
                print('shallow search move', move)
            state.makeMove(move[0],move[1],move[2],move[3])
        
def potentialMoves(state):
    moves = []
    hand = state.hands[state.activePlayer-1]
    #empty stations or stations with our coins on it
    availableStations = list(filter(lambda x: state.stationOwners[x] == 0 or state.stationOwners[x] == state.activePlayer, range(11)))
    #stations with their coins on it
    theirStations = list(set(range(11)) - set(availableStations)) 
    #available stations with coins
    availableStationsWC = list(filter(lambda x: state.stationOwners[x] == state.activePlayer, range(11)))
    #stations with 1 coin on it that can be taken by removing the coin with the premove
    flippableStations = list(filter(lambda x: state.stationCounts == 1, theirStations))
    if 0 in hand:
        for as1 in availableStations:
            moves.append([0,as1,-1,-1,-1])
        for fs in flippableStations:
            moves.append([[0,fs,-1,-1,fs]])
    if 1 in hand:
        for as1 in availableStations:
            for as2 in availableStations:
                moves.append([1,as1,as2,-1,-1])
        for fs in flippableStations:
            for s in availableStations+[fs]:
                moves.append([1,fs,s,-1,fs])
    if 2 in hand:
        for switch in range(10):
            moves.append([2,-1,-1,switch,-1])
    if 3 in hand:
        for station in range(11):
            if station in availableStationsWC:
                for as1 in availableStations:
                    moves.append([3,station,as1,-1,-1])
                for fs in flippableStations:
                    moves.append([3,station,fs,-1,fs])
            elif station in theirStations:
                for ts in theirStations:
                    if not ts == station:
                        moves.append([3,station,ts,-1,-1]) 
    return moves

#from a list of moves and score outcomes for those moves, find the most rewarding winning move or a tied move or the least bad losing move
def optimalMove(moves, scores,activePlayer,otherPlayer,premoveBonus):
    bestMove = -1
    winningMoves = list(filter(lambda x: (scores[x][activePlayer]+premoveBonus[x][activePlayer]) > (scores[x][otherPlayer]+premoveBonus[x][otherPlayer]), range(len(moves))))
    if winningMoves:
        highest = 0
        highestIndex = 0
        for i in range(len(winningMoves)):
            points = sum(scores[winningMoves[i]])
            if points > highest:
                highest = points
                highestIndex = i
        bestMove = winningMoves[highestIndex]
    else:
        tiedMoves = list(filter(lambda x: (scores[x][activePlayer]+premoveBonus[x][activePlayer]) == (scores[x][otherPlayer]+premoveBonus[x][activePlayer]),range(len(moves))))
        if tiedMoves:
            bestMove = tiedMoves[0]
        else:
            lowest = 1000
            lowestIndex = 0
            for i in range(len(moves)):
                points = sum(scores[i])
                if points < lowest:
                    lowest = points
                    lowestIndex = i
            bestMove = lowestIndex
    assert 0 <= bestMove and bestMove < len(moves)
    return bestMove

#overlapMultiplier. recommended between 0 and 1. How much the AI cares about stations being part of multiple cycles
#fortifyMultiplier. recommended between 0 and 1. Reduces station score when it already has a lot of coins
#emptyBonus. Recommended between 0 and 2. Causes the AI to capture the best empty station
#stationMultipliers. Recommended between 0.5 and 1.5. Preference for individual stations
#spreadTreshold. Recommended between 0 and 2. Spread coins over multiple stations if the best stations are close in score
def ai1(state, overlapMultiplier=0.1, fortifyMultiplier = 0.1, emptyBonus = 1, spreadTreshold = 1, stationMultipliers1 = [1]*11, stationMultipliers2 = [1]*11):
    ### start by giving a score to each station which indicates how good the AI rates that station
    ### based on the cycles each station is part of and how many switches need to be changed to reach those cycles
    scores = [0,0,0,0,0,0,0,0,0,0,0]
    #first make a list for each station. It contains the switch distances of each cycle that station is a part of
    stationDistances = [[],[],[],[],[],[],[],[],[],[],[]]
    for i,c in enumerate(state.cycles):
        for s in c:
            stationDistances[s].append(state.switchDistances[i])
    
    #for each cycle, calculate an initial score based on the best cycle it is currently part of
    #also increase the score slightly based on the switchdistances of other cycles it is a part of
    for i in range(11):
        stationDistances[i].sort()
        stationDistC = list(stationDistances[i])
        while stationDistC:
            distance = stationDistC.pop()
            multiplier = 1
            if scores[i] > 0:
                multiplier = overlapMultiplier
            scores[i] += multiplier * (7 - distance)
        if state.stationCounts[i] == 0:
            scores[i] += emptyBonus
        else:
            scores[i] -= state.stationCounts[i] * fortifyMultiplier
            
        if state.game == 0:
            scores[i] *= stationMultipliers1[i]
        elif state.game == 1:
            scores[i] *= stationMultipliers2[i]
        else:
            print('error invalid game')
        
    topStation = scores.index(max(scores))
    state.makePreMove(topStation)
    
    #recalculate score of premove station
    score = 0
    while stationDistances[topStation]:
        distance = stationDistances[topStation].pop()
        multiplier = 1
        if score > 0:
            multiplier = overlapMultiplier
        score += multiplier * (7 - distance)
    if state.stationCounts[i] == 0:
        score += emptyBonus
    else:
        score -= state.stationCounts[topStation] * fortifyMultiplier
    if state.game == 0:
        score *= stationMultipliers1[topStation]
    elif state.game == 1:
        score *= stationMultipliers2[topStation]
    scores[topStation] = score
    
    hand = state.hands[state.activePlayer-1]
    
    #if the deck is empty
    if not state.deck:
        #if state space is too big
        if (len(state.hands[0])+len(state.hands[1])) > 3:
            #calculate one turn by thinking n turns ahead
            nTurnAI(state,first=True,n=1,usedPremove=state.usedPremove)
        else: #state space is managable
            #calculate best moves for both players
            perfectAI(state, first=True,usedPremove=state.usedPremove)
    
    #place coin cards in hand, get best two stations to place coins
    elif 0 in hand or 1 in hand:
        potentialStations = list(filter(lambda x: state.stationOwners[x] == 0 or state.stationOwners[x] == state.activePlayer, range(11)))
        station1 = potentialStations[0]
        station2 = potentialStations[0]
        for ps in potentialStations:
            if scores[ps] > scores[station1]:
                station2 = station1
                station1 = ps
            elif scores[ps] > scores[station2]:
                station2 = ps
        card = 0
        if (scores[station1] - scores[station2]) > spreadTreshold:
            station2 = station1
            if not 0 in hand:
                card = 1
        elif 1 in hand:
            card = 1
        if verbose:
            state.printState()
            print('move:',card,station1,station2)
        state.makeMove(card=card,station1=station1,station2=station2)
    else:
        nTurnAI(state,first=True,n=0,usedPremove=True)
        state.usedPremove=True
            
class State:
    def init(self, game):
        self.switches = [True,True,True,True,True,True,True,True,True,True] #each element represents a switch. True=default direction
        self.stationCounts = [0,0,0,0,0,0,0,0,0,0,0] #each element represents a station. The number is the number of coins
        self.stationOwners = [0,0,0,0,0,0,0,0,0,0,0] #each element represents a station. The number is the player who owns it. 0=nobody
        #the stations on each cycle
        self.cycles = [[5,6,9],[5,7,10],[4,5,6,7,10],[4,5,6,7,8,9,10],[4,7,8,10],[8],[0,1],[0,1,2],[2],[3],[2,3]]
        #the sums of points both players have on each cycle
        self.cyclesPoints = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        self.switchDistances = [0,0,0,0,0,0,0,0,0,0,0] #the minimum number of switches that need to be changed to achieve each cycle
        self.activePlayer = game%2+1
        self.game = game #which round is currently being played.
        self.totalPlayers = 2
        self.scores = [0,0]
        self.deck = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3]
                        #0 = place 3 coins
                        #1 = distribute 2 coins
                        #2 = change switch
                        #3 = move coins
        self.hands = [[],[]] #hands[0] is the first players hand, hands[1] is the second players hand, etc
        self.premoveBonus = [0,0]
        self.usedPremove = False
        self.calculateSwitchDistance()
        self.shuffle()
        self.dealHands()
            
    #this function calculates the minimum amount of switches that need to be changed in order for each cycle to become the final cycle
    #this 'switch distance' is a distance between game states and is used by the AI to find good moves
    def calculateSwitchDistance(self):
        switches = self.switches
        if (self.game%2) == 0:
            self.switchDistances[0] = (0 if not switches[7] else 1) + (0 if switches[9] else 1) + (0 if not switches[8] or not switches[6] or not switches[5] else 1)
            alt = (0 if switches[7] else 1) + (0 if switches[0] else 1) + (0 if not switches[1] else 1) + (0 if switches[4] else 1) + (0 if switches[9] else 1) + (0 if not switches[5] else 1)
            if alt < self.switchDistances[0]:
                self.switchDistances[0] = alt
            self.switchDistances[1] = (0 if not switches[9] else 1) + (0 if not switches[6] else 1) + (0 if not switches[7] or (switches[0] and not switches[1] and switches[4] and not switches[5]) else 1)
            self.switchDistances[2] = (0 if not switches[9] else 1) + (0 if not switches[5] else 1) + (0 if switches[6] else 1) + (0 if not switches[7] or (switches[0] and not switches[1] and switches[4]) else 1)
            self.switchDistances[3] = (0 if not switches[7] else 1) + (0 if not switches[8] else 1) + (0 if not switches[9] else 1) + (0 if switches[6] else 1) + (0 if switches[5] else 1)
            self.switchDistances[4] = (0 if not switches[7] else 1) + (0 if switches[8] else 1) + (0 if switches[5] else 1) + (0 if switches[6] else 1)
            self.switchDistances[5] = (0 if switches[7] else 1) + (0 if switches[0] else 1) + (0 if not switches[1] else 1) + (0 if switches[4] else 1) + (0 if switches[5] else 1)
            self.switchDistances[6] = (0 if switches[7] else 1) + (0 if not switches[0] else 1)
            self.switchDistances[7] = (0 if switches[7] else 1) + (0 if switches[0] else 1) + (0 if switches[1] else 1) + (0 if switches[2] else 1)
            self.switchDistances[8] = (0 if switches[7] else 1) + (0 if switches[0] else 1) + (0 if switches[1] else 1) + (0 if not switches[2] else 1) + (0 if not switches[3] else 1)
            self.switchDistances[9] = (0 if switches[7] else 1) + (0 if switches[0] else 1) + (0 if not switches[1] else 1) + (0 if not switches[4] else 1)
            self.switchDistances[10] = (0 if switches[7] else 1) + (0 if switches[0] else 1) + (0 if switches[1] else 1) + (0 if not switches[2] else 1) + (0 if switches[3] else 1)
        else:
            self.switchDistances[0] = (0 if not switches[1] else 1) + (0 if switches[4] else 1) + (0 if switches[9] else 1) + (0 if not switches[5] or (not switches[7] and (not switches[8] or not switches[6])) else 1)
            self.switchDistances[1] = (0 if not switches[1] else 1) + (0 if switches[4] else 1) + (0 if not switches[9] else 1) + (0 if not switches[6] else 1) + (0 if not switches[5] or not switches[7] else 1)
            self.switchDistances[2] = (0 if not switches[1] else 1) + (0 if switches[4] else 1) + (0 if switches[6] else 1) + (0 if not switches[5] else 1) + (0 if not switches[9] else 1) + (0 if switches[6] else 1)
            self.switchDistances[3] = (0 if not switches[7] else 1) + (0 if not switches[8] else 1) + (0 if not switches[9] else 1) + (0 if switches[6] else 1) + (0 if switches[5] else 1) + (0 if not switches[1] else 1) + (0 if switches[4] else 1)
            self.switchDistances[4] = (0 if not switches[7] else 1) + (0 if switches[8] else 1) + (0 if switches[5] else 1) +  (0 if switches[6] else 1) + (0 if not switches[1] else 1) + (0 if switches[4] else 1)
            self.switchDistances[5] = (0 if switches[7] else 1) + (0 if switches[0] else 1) + (0 if not switches[1] else 1) + (0 if switches[4] else 1) + (0 if switches[5] else 1)
            self.switchDistances[6] = (0 if not switches[0] else 1) + (0 if switches[1] else 1) + (0 if switches[2] else 1)
            alt = (0 if not switches[0] else 1) + (0 if not switches[1] else 1) + (0 if switches[4] else 1) + (0 if switches[7] else 1) + (0 if switches[5] else 1)
            if alt < self.switchDistances[6]:
                self.switchDistances[6] = alt
            self.switchDistances[7] = (0 if switches[0] else 1) + (0 if switches[1] else 1) + (0 if switches[2] else 1)
            self.switchDistances[8] = (0 if switches[1] else 1) + (0 if not switches[2] else 1) + (0 if not switches[3] else 1)
            self.switchDistances[9] = (0 if not switches[1] else 1) + (0 if not switches[4] else 1)
            self.switchDistances[10] = (0 if switches[1] else 1) + (0 if not switches[2] else 1) + (0 if switches[3] else 1)            
        #assert there is exactly one cycle with switch distance 0
        assert len(list(filter(lambda x: x == 0, self.switchDistances))) == 1
    
    def calculateCycleTotals(self):
        for i in range(len(self.cycles)): #for each cycle
            stations = self.cycles[i] #get the stations
            #split the stations into a list for both players
            stations1 = list(filter(lambda x: self.stationOwners[i] == 1, stations))
            stations2 = list(filter(lambda x: self.stationOwners[i] == 2, stations))
            #store the sum of points for each player for the stations they own
            self.cyclesPoints[i][0] = sum(list(map(lambda x: self.stationCounts[x], stations1)))
            self.cyclesPoints[i][1] = sum(list(map(lambda x: self.stationCounts[x], stations2)))
            
    
    #randomize the deck
    def shuffle(self):
        random.seed()
        random.shuffle(self.deck)
    
    #deal 5 random cards to each player
    def dealHands(self):                           
        for player in range(self.totalPlayers):                #for each player
            while len(self.hands[player]) < 5:                 #loop until that players hand has 5 cards
                self.hands[player].append(self.deck.pop())     #move card from deck to players hand
        
    def makePreMove(self,station):
        assert 0 <= station and station < 11
        
        if self.stationOwners[station] == 0 or self.stationOwners[station] == self.activePlayer:
            self.stationOwners[station] = self.activePlayer
            self.stationCounts[station] += 1
        elif self.stationOwners[station] == ((self.activePlayer % self.totalPlayers) + 1):
            assert self.stationCounts[station] > 0
            self.stationCounts[station] -= 1
            if self.stationCounts[station] == 0:
                self.stationOwners[station] = 0
        else:
            print('error, invalid station owner')
        self.calculateCycleTotals()
        
    def makeMove(self, card, station1=-1,station2=-1,switch=-1):
        assert 0 <= card and card < 4
        assert card in self.hands[self.activePlayer-1]
        
        if card == 0: #place 3 coins on one station
            assert 0 <= station1 and station1 < 11
            assert self.activePlayer == self.stationOwners[station1] or 0 == self.stationOwners[station1] 
            self.stationCounts[station1] += 3
            self.stationOwners[station1] = self.activePlayer
        elif card == 1: #distribute 2 coins
            assert 0 <= station1 and station1 < 11
            assert self.activePlayer == self.stationOwners[station1] or 0 == self.stationOwners[station1]
            assert 0 <= station2 and station2 < 11
            assert self.activePlayer == self.stationOwners[station2] or 0 == self.stationOwners[station2]
            self.stationOwners[station1] = self.activePlayer
            self.stationOwners[station2] = self.activePlayer
            self.stationCounts[station1] += 1
            self.stationCounts[station2] += 1
        elif card == 2: #change switch 
            assert 0 <= switch and switch < 10
            self.switches[switch] = not self.switches[switch] #switch the switch
            self.calculateSwitchDistance()
        elif card == 3: #move coins
            assert 0 <= station1 and station1 < 11
            assert 0 <= station2 and station2 < 11
            assert self.stationOwners[station1] == self.stationOwners[station2] or self.stationOwners[station1] == 0 or self.stationOwners[station2] == 0
            self.stationCounts[station2] += self.stationCounts[station1]
            if self.stationCounts[station1] > 0:
                self.stationOwners[station2] = self.stationOwners[station1]
            self.stationCounts[station1] = 0
            self.stationOwners[station1] = 0
        self.hands[self.activePlayer-1].remove(card)
        if self.deck:
            self.hands[self.activePlayer-1].append(self.deck.pop())
        self.activePlayer = (self.activePlayer % self.totalPlayers) + 1 
        self.calculateCycleTotals()
        
    def printState(self):
        print('deck:',self.deck)
        print('hands',self.hands)
        print('cyclepoints',self.cyclesPoints)
        print('switchdistances', self.switchDistances)
        print('stationPoints',self.stationCounts)
        print('stationOwners',self.stationOwners)

#this function takes the switch configuration and returns the index of the final cycle
def getCycle(switches, startingStation):
    if startingStation == 0:
        if not switches[7] and switches[9] and (not switches[8] or not switches[6] or not switches[5]):
            return 0
        if switches[7] and switches[0] and not switches[1] and switches[4] and switches[9] and not switches[5]:
            return 0
        if not switches[9] and not switches[6] and (not switches[7] or (switches[0] and not switches[1] and switches[4] and not switches[5])):
            return 1
        if not switches[9] and not switches[5] and switches[6] and (not switches[7] or (switches[0] and not switches[1] and switches[4])):
            return 2
        if not switches[7] and not switches[8] and not switches[9] and switches[6] and switches[5]:
            return 3
        if not switches[7] and switches[8] and switches[5] and switches[6]:
            return 4
        if switches[7] and switches[0] and not switches[1] and switches[4] and switches[5]:
            return 5
        if switches[7] and not switches[0]:
            return 6
        if switches[7] and switches[0] and switches[1] and switches[2]:
            return 7
        if switches[7] and switches[0] and switches[1] and not switches[2] and not switches[3]:
            return 8
        if switches[7] and switches[0] and not switches[1] and not switches[4]:
            return 9
        if switches[7] and switches[0] and switches[1] and not switches[2] and switches[3]:
            return 10
    elif startingStation == 1:
        if not switches[1] and switches[4] and switches[9] and (not switches[5] or (not switches[7] and (not switches[8] or not switches[6]))):
            return 0
        if not switches[1] and switches[4] and not switches[9] and not switches[6] and (not switches[5] or not switches[7]): 
            return 1
        if not switches[1] and switches[4] and switches[6] and not switches[5] and not switches[9] and switches[6]:
            return 2
        if not switches[7] and not switches[8] and not switches[9] and switches[6] and switches[5] and not switches[1] and switches[4]:
            return 3
        if not switches[7] and switches[8] and switches[5] and switches[6] and not switches[1] and switches[4]:
            return 4
        if switches[7] and switches[0] and not switches[1] and switches[4] and switches[5]:
            return 5
        if not switches[0] and ((switches[1] and switches[2]) or (not switches[1] and switches[4] and switches[5] and switches[7])):
            return 6
        if switches[0] and switches[1] and switches[2]:
            return 7
        if switches[1] and not switches[2] and not switches[3]:
            return 8
        if not switches[1] and not switches[4]:
            return 9
        if switches[1] and not switches[2] and switches[3]:
            return 10

#make a random move for the given state
def randomAI(state):
    card = state.hands[state.activePlayer-1][0]
    state.makePreMove(random.choice(range(11)))
    station1 = -1
    station2 = -1
    switch   = -1
    if card == 0:
        station1 = random.choice(list(filter(lambda x: state.stationOwners[x] == 0 or state.stationOwners[x] == state.activePlayer, range(11))))
    elif card == 1:
        station1 = random.choice(list(filter(lambda x: state.stationOwners[x] == 0 or state.stationOwners[x] == state.activePlayer, range(11))))
        station2 = random.choice(list(filter(lambda x: state.stationOwners[x] == 0 or state.stationOwners[x] == state.activePlayer, range(11))))
    elif card == 2:
        switch = random.choice(range(len(state.switches)))
    elif card == 3:
        station1 = random.choice(range(len(state.stationCounts)))
        owner = state.stationOwners[station1]
        station2 = random.choice(list(filter(lambda x: state.stationOwners[x] == 0 or state.stationOwners[x] == owner, range(11))))
        
    state.makeMove(card, station1, station2, switch)
    

    
#play a game of 2 rounds between two AIs
def playGame(pars1, pars2, verbose=False, outcomes=[0,0]):
    #random.seed(sum(pars1[4])+sum(pars1[5]))
    random.seed()
    points1 = 0
    points2 = 0
    for i in range(2):
        #create a new game state instance
        state = State()
        #init (initialize variables, shuffle and calculate switch distances for cycles)
        state.init(i)
        #as long as a player has at least one card in hand
        while len(state.hands[0]) or len(state.hands[1]):
            if state.activePlayer == 1:
                ai1(state,pars1[0],pars1[1],pars1[2],pars1[3],pars1[4],pars1[5])
            elif state.activePlayer == 2:
                ai1(state,*pars2)
            else:
                print('error: invalid player')
        finalCycle = getCycle(state.switches, state.game%2)
        if state.cyclesPoints[finalCycle][0] > state.cyclesPoints[finalCycle][1]:
            points1 += sum(state.cyclesPoints[finalCycle])
            if verbose:
                print(ai1.__name__ ,'(1) wins round',i+1,'with', sum(state.cyclesPoints[finalCycle]), 'points')
        elif state.cyclesPoints[finalCycle][0] < state.cyclesPoints[finalCycle][1]:
            points2 += sum(state.cyclesPoints[finalCycle])
            if verbose:
                print(ai2.__name__,'(2) wins round',i+1,'with', sum(state.cyclesPoints[finalCycle]), 'points')
        else:
            points1 += sum(state.cyclesPoints[finalCycle])/2
            points2 += sum(state.cyclesPoints[finalCycle])/2            
            if verbose:
                print('The round is a tie with', sum(state.cyclesPoints[finalCycle])/2, 'points each')
    if points1 > points2:
        if verbose:
            print(ai1.__name__ ,'(1) wins the game with', points1, 'points')
    elif points2 > points1:
        if verbose:
            print(ai2.__name__ ,'(2) wins the game with', points2, 'points')
    else:
        if verbose:
            print('The game is a tie with', points1, 'points each')
    outcomes[0]=points1
    outcomes[1]=points2
    
    
def nested_sum(L):
    total = 0  # don't use `sum` as a variable name
    for i in L:
        if isinstance(i, list):  # checks if `i` is a list
            total += nested_sum(i)
        else:
            total += i
    return total
