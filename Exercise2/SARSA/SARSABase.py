#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
from collections import defaultdict

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()

		self.S = [(x,y) for x in range(5) for y in range(6)]
		self.S.append("GOAL")
		self.S.append("OUT_OF_BOUNDS")

		self.discountFactor = discountFactor
		self.setEpsilon(epsilon)
		self.setLearningRate(learningRate)
		self.Q = defaultdict(float)
		self.policy = {s:"DRIBBLE_RIGHT" for s in self.S}
		self.experiences = []
		self.curState = (1, 1) # arbitrary

	def learn(self):

		s, a, r, sP = self.experiences.pop(0)
		aP = self.experiences[0][1]

		before = self.Q[(s, a)]
		self.Q[(s, a)] = self.Q[(s, a)] + self.learningRate*(r + self.discountFactor*self.Q[(sP, aP)] - self.Q[(s, a)])

		return self.Q[(s, a)] - before

			
	def act(self): # Use epsilon-greedy
		best = "DRIBBLE_RIGHT"; val = -10
		for newAction in self.possibleActions:
			if self.Q[(self.curState, newAction)] > val:
				val = self.Q[(self.curState, newAction)]
				best = newAction
		if np.random.random() < (1 - self.epsilon + self.epsilon/len(self.possibleActions)):
			return best
		else:
			return np.random.choice([a for a in self.possibleActions if a != best])

	def setState(self, state):
		self.curState = state

	def setExperience(self, state, action, reward, status, nextState):
		self.experiences.append( (state, action, reward, nextState) )

	def computeHyperparameters(self, numTakenActions, episodeNumber):

		# returned as learningRate, epsilon
		return max((500-episodeNumber)/1000, 0.1), max((500-episodeNumber)/500, 0.1)
		#return 0.5, 0.1

	def toStateRepresentation(self, state):
		# State comes in as the player's position and the opponent position
		# We are guaranteed that the opponent will not move so we just need
		# the first position
		return state[0]

	def reset(self):
		self.experiences = []

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99, 0.1)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	
