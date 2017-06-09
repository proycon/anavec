#!/usr/bin/env python3
import sys
import os
import argparse
import theano
import theano.tensor as T
import numpy as np
import colibricore


def randomMatrix(n, f):
    return np.random.randint(6, size=n*f).astype(np.int8).reshape((n, f))


def compute(trainingdata, testdata):
    # adapted from https://gist.github.com/danielvarga/d0eeacea92e65b19188c
    # with lamblin's workaround at https://github.com/Theano/Theano/issues/1399

    n = 10 # number of candidates
    m = 3 # number of targets
    f = 50  # number of features

    x = T.matrix('x') # candidates
    y = T.matrix('y') # targets

    xL2S = T.sum(x*x, axis=-1) # [n]
    yL2S = T.sum(y*y, axis=-1) # [m]
    xL2SM = T.zeros((m, n)) + xL2S # broadcasting, [m, n]
    yL2SM = T.zeros((n, m)) + yL2S # # broadcasting, [n, m]
    squaredPairwiseDistances = xL2SM.T + yL2SM - 2.0*T.dot(x, y.T) # [n, m]

    np.random.seed(1)

    N = randomMatrix(n, f)
    M = randomMatrix(m, f)

    lamblinsTrick = False

    if lamblinsTrick:
        s = squaredPairwiseDistances
        bestIndices = T.cast( ( T.arange(n).dimshuffle(0, 'x') * T.cast(T.eq(s, s.min(axis=0, keepdims=True)), 'float32') ).sum(axis=0), 'int32')
    else:
        bestIndices = T.argmin(squaredPairwiseDistances, axis=0)

    print("N", N)
    print("M", M)

    nearests_fn = theano.function([x, y], bestIndices, profile=False)

    return nearests_fn(trainingdata, testdata)

UNKFEATURE = -1
PUNCTFEATURE = -1

def buildfeaturevector(word, alphabetmap, numfeatures):
    featurevector = np.zeros(numfeatures, dtype=np.uint8)
    for char in word:
        if not char.isalnum():
            featurevector[PUNCTFEATURE] += 1
        elif char in alphabetmap:
            featurevector[alphabetmap[char]] += 1
        else:
            featurevector[UNKFEATURE] += 1
    return featurevector


def main():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--patternmodel', type=str,help="Pattern model of background corpus (training data)", action='store',default="",required=True)
    parser.add_argument('-c','--classfile', type=str,help="Class file of background corpus", action='store',default="",required=True)
    parser.add_argument('-u','--unkweight', type=float,help="Unknown character weight", action='store',default=1,required=True)
    parser.add_argument('-p','--puncweight', type=float,help="Punctuation character weight", action='store',default=1,required=True)
    parser.add_argument('-d', '--debug',action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.classfile):
        print("Error: Class file " + args.classfile + " does not exist",file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.patternmodel):
        print("Error: Pattern model file " + args.patternmodel + " does not exist",file=sys.stderr)
        sys.exit(2)

    print("Test input words, one per line (if interactively invoked, type ctrl-D when done)",file=sys.stderr)
    testwords = sys.stdin.readlines()

    numtest = len(testwords)
    print("Test set size: ", numtest, file=sys.stderr)

    classencoder= colibricore.ClassEncoder(args.classfile)
    classdecoder = colibricore.ClassDecoder(args.classfile)
    patternmodel = colibricore.UnindexedPatternModel(args.patternmodel)

    alphabet = {}

    numtraining = 0 #number of word types
    print("Computing alphabet on training data...",file=sys.stderr)
    for pattern in patternmodel:
        if len(pattern) == 1: #only unigrams for now
            numtraining += 1
            word = pattern.tostring(classdecoder) #string representation
            for char in word and char not in alphabet:
                if char.isalpha():
                    alphabet.add(char)


    #maps each character to a feature number
    alphabetmap = {}
    for i, char in enumerate(sorted(alphabet)):
        alphabetmap[char] = i

    print("Alphabet computed (size=" + str(len(alphabet))+"): ", alphabet, file=sys.stderr)
    numfeatures = len(alphabet) + 2 #UNK feature, PUNCT feature
    if args.debug: print(alphabetmap)

    print("Building test vectors...", file=sys.stderr)
    testdata = np.array( ( buildfeaturevector(testword, alphabetmap, numfeatures ) for testword in testwords) )
    if args.debug: print("[DEBUG] TEST DATA: ", testdata)

    print("Building training vectors...", file=sys.stderr)
    trainingdata = np.empty((numtraining, numfeatures), dtype=np.int8)

    index = 0
    for pattern in patternmodel:
        if len(pattern) == 1: #only unigrams for now
            trainingdata[index] = buildfeaturevector(pattern.tostring(classdecoder), alphabetmap, numfeatures)
            index  += 1

    if args.debug: print("[DEBUG] TRAINING DATA: ", trainingdata)

    result = compute(trainingdata, testdata)

if __name__ == '__main__':
    main()
