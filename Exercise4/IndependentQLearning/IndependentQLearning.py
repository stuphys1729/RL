#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np
from collections import defaultdict
		
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()

		self.discountFactor = discountFactor
		self.setEpsilon(epsilon)
		self.setLearningRate(learningRate)
		self.Q = defaultdict(float)
		self.experience = None
		self.curState = (1, 1) # arbitrary

	def setExperience(self, state, action, reward, status, nextState):
		self.experience = (state, action, reward, nextState)
	
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
		if state == "GOAL" or state == "OUT_OF_BOUNDS":
			return state
		# State comes in as state[0] being the positions of both agents
		# and state[2][0] being the position of the ball
		# and state[1][0] being the position of the defender which does not change
		return (tuple(state[0][0]), tuple(state[0][1]), tuple(state[2][0]))

	def setState(self, state):
		self.curState = state

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return max((5000-episodeNumber)/10000, 0.1), max((5000-episodeNumber)/5000, 0.1)
		#return 0.1, 0.1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	cumulativeReward = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			totalReward += reward[0]

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation

		cumulativeReward += totalReward
		if episode % 100 == 0:
			print(cumulativeReward, episode)

	print("Total Reward after {} episodes: {}".format(numEpisodes, cumulativeReward))
				
