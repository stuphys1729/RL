#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
from collections import defaultdict

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		
		self.S = [(x,y) for x in range(5) for y in range(6)]
		self.S.append("GOAL")
		self.S.append("OUT_OF_BOUNDS")

		self.discountFactor = discountFactor
		self.setEpsilon(epsilon)
		self.setLearningRate(learningRate)
		self.Q = defaultdict(float)
		self.policy = {s:"DRIBBLE_RIGHT" for s in self.S}
		self.experience = None
		self.curState = (1, 1) # arbitrary

	def learn(self):
		s, a, r, sP = self.experience
		before = self.Q[(s, a)]

		val = -10
		for action in self.possibleActions:
			val = max(val, self.Q[(sP, action)])

		self.Q[(s, a)] += self.learningRate * (r + self.discountFactor*val - self.Q[(s, a)])

		return self.Q[(s, a)] - before

	def act(self):
		best = "DRIBBLE_RIGHT"; val = -10
		for newAction in self.possibleActions:
			if self.Q[(self.curState, newAction)] > val:
				val = self.Q[(self.curState, newAction)]
				best = newAction

		if np.random.random() < (1 - self.epsilon + self.epsilon/len(self.possibleActions)):
			return best
		else:
			return np.random.choice([a for a in self.possibleActions if a != best])

	def toStateRepresentation(self, state):
		# State comes in as the player's position and the opponent position
		# We are guaranteed that the opponent will not move so we just need
		# the first position
		return state[0]

	def setState(self, state):
		self.curState = state

	def setExperience(self, state, action, reward, status, nextState):
		self.experience = (state, action, reward, nextState)

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def reset(self):
		self.experience = None
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# returned as learningRate, epsilon
		return max((500-episodeNumber)/1000, 0.1), max((500-episodeNumber)/500, 0.1)
		#return 0.1, 0.1

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	
