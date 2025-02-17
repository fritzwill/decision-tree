import pandas as pd 
import numpy as np 
import math

# Handle training data so it can be loaded once, then referenced from there
# Make any alteracations to number of features used in training data
# Read in original data and get subset table with columns:
## Is_Home_or_Away
## Is_Opponent_in_AP25_Preseason
## Label
DF_TRAIN = pd.read_csv('Dataset-football-train.txt',sep='\t')
DF_TRAIN = DF_TRAIN[['Is_Home_or_Away','Is_Opponent_in_AP25_Preseason','Media','Label']]


class Tree:
	def __init__(self,observationIDs,features,currLvl=0,subTree={},bestFeature=None,majorityLabel=None,parentMajorityLabel=None):
		self.observationIDs = observationIDs
		self.features = features
		self.currLvl = currLvl
		self.subTree = subTree
		self.bestFeature = bestFeature
		self.majorityLabel = majorityLabel
		self.parentMajorityLabel = parentMajorityLabel
		self.setBestFeatureID(bestFeature)

	# predicts using a tree and 
	# observation: [Is_Home_or_Away, Is_Opponent_in_AP25_Preseason, Media]

	def setBestFeatureID(self, feature):
		idx = None
		if feature == 'Is_Home_or_Away':
			idx = 0
		elif feature == 'Is_Opponent_in_AP25_Preseason':
			idx = 1
		else:
			idx = 2
		self.bestFeatureID = int(idx)

def predict(tree, obs):
	if tree.bestFeature == None:
		return tree.majorityLabel
	featVal = obs[tree.bestFeatureID]
	if not featVal in tree.subTree: # val with no subtree
		return tree.majorityLabel
	else: # recurse on subtree
		return predict(tree.subTree[featVal],obs)

def displayDecisionTree(tree):
	print('\t'*tree.currLvl + '(lvl {}) {}'.format(tree.currLvl,tree.majorityLabel))
	if tree.bestFeature == None:
		return

	print('\t'*tree.currLvl + '{}'.format(tree.bestFeature) + ': ')
	for [val,subTree] in sorted(tree.subTree.items()):
		print('\t'*(tree.currLvl+1) + 'choice: {}'.format(val))
		displayDecisionTree(subTree)

def Entropy(ns):
	entropy = 0.0
	total = sum(ns)
	for x in ns:
		entropy += -1.0*x/total*math.log(1.0*x/total,2)
	return entropy

# Information Gain
def IG(observationIDs, feature):
	# get smaller dataframe
	df = DF_TRAIN.loc[observationIDs]
	# populate counts for Wins/Losses for each category of the feature
	labelCountDict = {}
	valueLabelCountDict = {}
	for index, row in df.iterrows():
		label = row['Label']
		if not label in labelCountDict:
			labelCountDict[label] = 0 # this specific label was not found so insert 0 count
		labelCountDict[label] += 1
		featureValue = row[feature]
		if not featureValue in valueLabelCountDict:
			valueLabelCountDict[featureValue] = {} # this specific feature value not found so insert empty dict
		if not label in valueLabelCountDict[featureValue]:
			valueLabelCountDict[featureValue][label] = 0 # this specific label was not found for this feature value so insert 0 count
		valueLabelCountDict[featureValue][label] += 1

	ns = []
	for [label,count] in labelCountDict.items():
		ns.append(count)

	H_Y = Entropy(ns)

	H_Y_X = 0.0
	for [featureValue, labelCountDict] in valueLabelCountDict.items():
		nsHYX = []
		for [label,count] in labelCountDict.items():
			nsHYX.append(count)
		H_Y_X += 1.0*sum(nsHYX)/len(df)*Entropy(nsHYX)
	return H_Y - H_Y_X

def GR(observationIDs, feature):
	ig = IG(observationIDs,feature)
	if ig == 0:
		return 0
	df = DF_TRAIN.loc[observationIDs]
	valueLabelDict = {}
	for index, row in df.iterrows():
		label = row['Label']
		featureValue = row[feature]
		if featureValue not in valueLabelDict:
			valueLabelDict[featureValue] = 0
		valueLabelDict[featureValue] += 1
	ns = []
	for [val,count] in valueLabelDict.items():
		ns.append(count)
	ent = Entropy(ns)
	return float(ig)/ent

def fillDecisionTree(tree,decisionTreeAlgo):
	# find the majorityLabel
	df = DF_TRAIN.loc[tree.observationIDs] # smaller df
	counts = df['Label'].value_counts()
	majorityLabel = df['Label'].value_counts().idxmax()
	if len(counts) > 1:
		if counts['Win'] == counts['Lose']:
			majorityLabel = tree.parentMajorityLabel
	tree.majorityLabel = majorityLabel

	# exit if only one label
	if len(counts) == 1:
		return
	# exit if no features left
	if len(tree.features) == 0: 
		return

	# find best feature
	featureValueDict = {}
	for feature in tree.features: 
		if decisionTreeAlgo == 'ID3':
			metricScore = IG(tree.observationIDs,feature)
		if decisionTreeAlgo == 'C45':
			metricScore = GR(tree.observationIDs,feature)
		featureValueDict[feature] = metricScore
	bestFeature, bestFeatureValue = sorted(featureValueDict.items(),reverse=True)[0]
	# exit if IG or GR is 0
	if bestFeatureValue == 0.0:
		return
	tree.bestFeature = bestFeature

	# find subset of features
	subFeatures = set()
	for feature in tree.features:
		if feature == bestFeature: # skip the current best feature
			continue
		subFeatures.add(feature)
	
	# find best feature id
	bestFeatureIdx = 0
	if bestFeature == 'Is_Home_or_Away':
		bestFeatureIdx = 0
	elif bestFeature == 'Is_Opponent_in_AP25_Preseason':
		bestFeatureIdx = 1
	else:
		bestFeatureIdx = 2
	
	# find subset of observations
	subObservationsDict = {}
	for obs in tree.observationIDs:
		val = DF_TRAIN.values[obs][bestFeatureIdx]
		if not val in subObservationsDict:
			subObservationsDict[val] = set()
		subObservationsDict[val].add(obs)

	for [val,obs] in subObservationsDict.items():

		tree.subTree[val] = Tree(obs, subFeatures, tree.currLvl + 1,{},None,None,majorityLabel)
		
		fillDecisionTree(tree.subTree[val],decisionTreeAlgo)

def predictAndAnalyze(tree, data):
	TP = 0
	FN = 0
	FP = 0
	TN = 0
	for obs in data:
		prediction = predict(tree,obs)
		ground = obs[3]
		if prediction == 'Win' and ground == 'Win':
			TP += 1
		if prediction == 'Win' and ground == 'Lose':
			FP += 1
		if prediction == 'Lose' and ground == 'Win':
			FN += 1
		if prediction == 'Lose' and ground == 'Lose':
			TN += 1

	accuracy = float(TP+TN)/len(data)
	precision = float(TP)/(TP + FP)
	recall = float(TP)/(TP + FN)
	F1 = 2*(recall*precision)/(recall+precision)
	print('\nAnalysis:')
	print('accuracy = {}'.format(accuracy))
	print('precision = {}'.format(precision))
	print('recall = {}'.format(recall))
	print('F1 score = {}'.format(F1))


# read in original data and get subset table with columns:
## Is_Home_or_Away
## Is_Opponent_in_AP25_Preseason
## Label
dfTest = pd.read_csv('Dataset-football-test.txt',sep='\t')
dfTest = dfTest[['Is_Home_or_Away','Is_Opponent_in_AP25_Preseason','Media','Label']]

# obsIDs, features, lvl subTree, bestFeature, majority label, parent majority label
initialObservationIDs = set(range(len(DF_TRAIN)))
initialFeatures = set(dfTest.columns.values[:-1])

# prompt user
print("Which decision tree algorithm would you like to use ('ID3' or 'C45)?")
algoChoice = str(raw_input())
if algoChoice not in {'ID3','C45'}:
	print("Invalid algorithm choice. You must choose 'ID3' or 'C45'")
	exit()

print("choice: {}".format(algoChoice))

MyTree = Tree(initialObservationIDs,initialFeatures)
fillDecisionTree(MyTree,algoChoice)

print('My Decision Tree:')
displayDecisionTree(MyTree)


print('Predicted Labels of Test Data:')
predictAndAnalyze(MyTree,dfTest.values)

