#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
from collections import defaultdict
import numpy as np

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()

		self.S = [(x,y) for x in range(5) for y in range(6)]
		self.S.append("GOAL")
		self.S.append("OUT")

		self.discountFactor = discountFactor
		self.setEpsilon(epsilon)
		self.Q = defaultdict(float)
		self.returns = defaultdict(list)
		self.policy = {s:"DRIBBLE_RIGHT" for s in self.S}
		self.visits = []
		self.experiences = []
		self.curState = (-1, -1) # So this will error out obviously if it bleeds through

	def learn(self):
		G = 0
		updates = []
		visits = set()
		while len(self.experiences) != 0:
			state, action = self.visits.pop()
			reward, status, nextState = self.experiences.pop()
			G = self.discountFactor*G + reward
			if (state, action) not in self.visits:
				visits.add( (state, action) )
				self.returns[(state, action)].append(G)
				self.Q[(state, action)] = np.mean(self.returns[(state, action)])

				best = "DRIBBLE_UP"; val = -10
				for newAction in self.possibleActions:
					if self.Q[(state, newAction)] > val:
						val = self.Q[(state, newAction)]
						best = newAction

				self.policy[state] = best
				updates.append(self.Q[(state, action)])

		return (self.Q, updates[::-1])

	def toStateRepresentation(self, state):
		# State comes in as the player's position and the opponent position
		# We are guaranteed that the opponent will not move so we just need
		# the first position
		return state[0]

	def setExperience(self, state, action, reward, status, nextState):
		self.visits.append( (state, action) )
		self.experiences.append( (reward, status, nextState) )

	def setState(self, state):
		self.curState = state

	def reset(self):
		self.visits = []
		self.experiences = []
		self.curState = (-1, -1)

	def act(self):
		greedy = self.policy[self.curState]
		if np.random.random() < (1 - self.epsilon + self.epsilon/len(self.possibleActions)):
			return greedy
		else:
			return np.random.choice([a for a in self.possibleActions if a != greedy])

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return 0.1 # Not sure how to change this as of yet


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()
