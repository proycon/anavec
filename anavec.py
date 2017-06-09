#!/usr/bin/env python3
import sys
import os
import argparse
from collections import defaultdict
import theano
import theano.tensor as T
import numpy as np
import Levenshtein
import colibricore


UNKFEATURE = -1
PUNCTFEATURE = -2


def compute(trainingdata, testdata):
    # adapted from https://gist.github.com/danielvarga/d0eeacea92e65b19188c
    # with lamblin's workaround at https://github.com/Theano/Theano/issues/1399

    n = trainingdata.shape[0] # number of candidates
    assert testdata.shape[1] == trainingdata.shape[1]
    m = testdata.shape[0] # number of targets
    f = testdata.shape[1] # number of features

    x = T.matrix('x') # candidates
    y = T.matrix('y') # targets

    xL2S = T.sum(x*x, axis=-1) # [n]
    yL2S = T.sum(y*y, axis=-1) # [m]
    xL2SM = T.zeros((m, n)) + xL2S # broadcasting, [m, n]
    yL2SM = T.zeros((n, m)) + yL2S # # broadcasting, [n, m]
    squaredPairwiseDistances = xL2SM.T + yL2SM - 2.0*T.dot(x, y.T) # [n, m]

    lamblinsTrick = False

    if lamblinsTrick:
        s = squaredPairwiseDistances
        bestIndices = T.cast( ( T.arange(n).dimshuffle(0, 'x') * T.cast(T.eq(s, s.min(axis=0, keepdims=True)), 'float32') ).sum(axis=0), 'int32')
    else:
        bestIndices = T.argmin(squaredPairwiseDistances, axis=0)

    nearests_fn = theano.function([x, y], bestIndices, profile=False)

    return nearests_fn(trainingdata, testdata)


def buildfeaturevector(word, alphabetmap, numfeatures, args):
    featurevector = np.zeros(numfeatures, dtype=np.uint8)
    for char in word:
        if not char.isalnum():
            featurevector[PUNCTFEATURE] += 1 * args.punctweight
        elif char in alphabetmap:
            featurevector[alphabetmap[char]] += 1
        else:
            featurevector[UNKFEATURE] += 1 * args.unkweight
    return featurevector


def anahash(word, alphabetmap, numfeatures):
    hashvalue = 0
    for char in word:
        if not char.isalnum():
            charvalue = 100 + numfeatures + PUNCTFEATURE
        elif char in alphabetmap:
            charvalue = 100 + alphabetmap[char]
        else:
            charvalue = 100 + numfeatures + UNKFEATURE
        hashvalue += charvalue**5
    return hashvalue

def anahash_fromvector(vector):
    hashvalue = 0
    for i, count in enumerate(vector):
        hashvalue += count * ((100+i)**5)
    return hashvalue


def main():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--patternmodel', type=str,help="Pattern model of background corpus (training data)", action='store',default="",required=True)
    parser.add_argument('-c','--classfile', type=str,help="Class file of background corpus", action='store',default="",required=True)
    parser.add_argument('-u','--unkweight', type=float,help="Unknown character weight", action='store',default=1,required=False)
    parser.add_argument('-p','--punctweight', type=float,help="Punctuation character weight", action='store',default=1,required=False)
    parser.add_argument('-d', '--debug',action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.classfile):
        print("Error: Class file " + args.classfile + " does not exist",file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.patternmodel):
        print("Error: Pattern model file " + args.patternmodel + " does not exist",file=sys.stderr)
        sys.exit(2)

    print("Test input words, one per line (if interactively invoked, type ctrl-D when done)",file=sys.stderr)
    testwords = [ w.strip() for w in sys.stdin.readlines() ]

    numtest = len(testwords)
    print("Test set size: ", numtest, file=sys.stderr)

    classencoder= colibricore.ClassEncoder(args.classfile)
    classdecoder = colibricore.ClassDecoder(args.classfile)
    patternmodel = colibricore.UnindexedPatternModel(args.patternmodel)

    alphabet = set()


    numtraining = 0 #number of word types
    print("Computing alphabet on training data...",file=sys.stderr)
    for pattern in patternmodel:
        if len(pattern) == 1: #only unigrams for now
            numtraining += 1
            word = pattern.tostring(classdecoder) #string representation
            for char in word:
                if char.isalpha() and char not in alphabet:
                    alphabet.add(char)


    #maps each character to a feature number (order number)
    alphabetmap = {}
    for i, char in enumerate(sorted(alphabet)):
        alphabetmap[char] = i

    print("Alphabet computed (size=" + str(len(alphabet))+"): ", alphabet, file=sys.stderr)
    numfeatures = len(alphabet) + 2 #UNK feature, PUNCT feature
    if args.debug: print(alphabetmap)

    print("Building test vectors...", file=sys.stderr)
    testdata = np.array( [ buildfeaturevector(testword, alphabetmap, numfeatures, args ) for testword in testwords] )
    if args.debug: print("[DEBUG] TEST DATA: ", testdata)

    print("Building training vectors and counting anagram hashes...", file=sys.stderr)
    trainingdata = np.empty((numtraining, numfeatures), dtype=np.int8)


    anahashcount = defaultdict(int) #frequency count of all seen anahashes (uses to determine which are actual anagrams in the training data)

    instanceindex = 0
    for pattern in patternmodel:
        if len(pattern) == 1: #only unigrams for now
            word = pattern.tostring(classdecoder)
            h = anahash(word, alphabetmap, numfeatures)
            anahashcount[h] += 1
            trainingdata[instanceindex] = buildfeaturevector(word, alphabetmap, numfeatures, args) #TODO: words that are anagrams are duplicated in training data now, doesn't hurt but sligtly less efficient, but otherwise precomputed matrix size is has gaps
            instanceindex  += 1

    if args.debug: print("[DEBUG] TRAINING DATA DIMENSIONS: ", trainingdata.shape)

    print("Computing matches...", file=sys.stderr)
    bestindices = compute(trainingdata, testdata)

    print("Resolving results...", file=sys.stderr)

    matchinganagramhashes = defaultdict(set) #map of matching anagram hash to test words that yield it as a match
    for bestindex, testword in zip(bestindices, testwords):
        h = anahash_fromvector(trainingdata[bestindex])
        matchinganagramhashes[h].add(testword)

    solutions = defaultdict(list) #maps test words to solutions (the best match)

    for pattern in patternmodel:
        if len(pattern) == 1: #only unigrams for now
            trainingword = pattern.tostring(classdecoder)
            h = anahash(trainingword, alphabetmap, numfeatures)
            if h in matchinganagramhashes:
                for testword in matchinganagramhashes[h]:
                    solutions[testword].append(trainingword)

    print("Consolidating and outputting...", file=sys.stderr)
    #output in same order as input
    for testword in testwords:
        if len(solutions[testword]) == 1:
            #only one solution
            print(testword + "\t" + solutions[testword][0] + "\t1.0")
        else:
            #we have multiple solutions; we are going to use sources to disambiguate:
            #   1) the levensthein distance
            #   2) the frequency in the training corpus
            solutions_extended = [ (solution, Levenshtein.distance(testword, solution), patternmodel[classencoder.buildpattern(solution)]) for solution in solutions[testword] ]
            distancesum = sum((  distance for _, distance, _ in solutions_extended ))
            freqsum = sum(( freq for _, _, freq in solutions_extended ))
            #compute a normalize compound score including both components:
            solutions_scored = [ ( solution, ((distance / distancesum) + (freq / freqsum)) / 2.0 ) for solution, distance, freq in solutions_extended ]

            #output solutions:
            print(testword, end="")
            for solution, score in sorted(solutions_scored, key=lambda x: -1 * x[1]):
                print("\t" + solution + "\t" + str(score),end="")
            print()

if __name__ == '__main__':
    main()
