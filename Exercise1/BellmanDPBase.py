from MDP import MDP
import sys

class BellmanDPSolver(object):
    def __init__(self, discountRate=0.1):
        self.MDP = MDP()
        self.discountRate = discountRate
        self.initVs()

    def initVs(self):
        self.Vs = {s:0. for s in self.MDP.S}
        self.policy = {s:["DRIBBLE_RIGHT"] for s in self.MDP.S}

    def actionValue(self, state, action):
        actionVal = 0
        nextStates = self.MDP.probNextStates(state, action)
        for sPrime in nextStates:
            reward = self.MDP.getRewards(state, action, sPrime)
            actionVal += nextStates[sPrime] * (reward + self.discountRate*self.Vs[sPrime])
        return actionVal
    

    def BellmanUpdate(self):

        ## Update Value Function ##
        newVs = {}
        for state in self.MDP.S:
            newVal = -10
            for action in self.MDP.A:
                actionVal = self.actionValue(state, action)
                newVal = max(newVal, actionVal)
            
            newVs[state] = newVal
        self.Vs = newVs

        ## Update Greedy Policy ##
        for state in self.MDP.S:
            actionDict = {}
            maxVal = -10
            for action in self.MDP.A:
                actionVal = self.actionValue(state, action)
                maxVal = max(maxVal, actionVal)
                actionDict[action] = actionVal
            
            resList = []
            for action in self.MDP.A: # Ensures correct ordering
                if actionDict[action] == maxVal:
                    resList.append(action)
            
            self.policy[state] = resList
        
        return self.Vs, self.policy



if __name__ == '__main__':
    solution = BellmanDPSolver()
    for i in range(20000):
        values, policy = solution.BellmanUpdate()
    print("Values : ", values)
    print("Policy : ", policy)