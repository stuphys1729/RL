#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np
from collections import defaultdict
		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()

		self.discountFactor = discountFactor
		self.setEpsilon(epsilon)
		self.setLearningRate(learningRate)
		self.Q = defaultdict(float)
		self.experience = None
		self.curState = (1, 1) # arbitrary
		self.C = defaultdict(int)
		self.nS = 1 # stops division by zero

	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		self.experience = (state, action, oppoActions[0], reward, nextState)
		
	def learn(self):
		s, myA, tA, r, sP = self.experience
		before = self.Q[(s, myA, tA)]

		val = -10
		for myAction in self.possibleActions:
			actionVal = 0
			for theirAction in self.possibleActions:
				prob = self.C[sP, theirAction] / self.nS
				actionVal += prob * self.Q[(sP, myAction, theirAction)]
			val = max(val, actionVal)

		self.Q[(s, myA, tA)] += self.learningRate * (r + self.discountFactor*val - self.Q[(s, myA, tA)])

		self.C[s, tA] += 1
		self.nS += 1

		return self.Q[(s, myA, tA)] - before

	def act(self):
		best = "DRIBBLE_RIGHT"; val = -10
		for myAction in self.possibleActions:

			actionVal = 0
			for theirAction in self.possibleActions:
				prob = self.C[self.curState, theirAction] / self.nS
				actionVal += prob * self.Q[(self.curState, myAction, theirAction)]

			if actionVal > val:
				val = actionVal
				best = myAction
		
		if np.random.random() < (1 - self.epsilon + self.epsilon/len(self.possibleActions)):
			return best
		else:
			return np.random.choice([a for a in self.possibleActions if a != best])

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate) :
		self.learningRate = learningRate

	def setState(self, state):
		self.curState = state

	def toStateRepresentation(self, rawState):
		if rawState == "GOAL" or rawState == "OUT_OF_BOUNDS":
			return rawState
		# rawState comes in as rawState[0] being the positions of both agents
		# and rawState[2][0] being the position of the ball
		# and rawState[1][0] being the position of the defender which does not change
		try:
			state = (tuple(rawState[0][0]), tuple(rawState[0][1]), tuple(rawState[2][0]))
		except IndexError as e:
			print(state)
			raise e
		return state
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return max((5000-episodeNumber)/10000, 0.1), max((5000-episodeNumber)/5000, 0.1)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0
	cumulativeReward = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
			
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
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1
			totalReward += reward[0]

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation
		
		cumulativeReward += totalReward
		if episode % 100 == 0:
			print(cumulativeReward, episode)

	print("Total Reward after {} episodes: {}".format(numEpisodes, cumulativeReward))
