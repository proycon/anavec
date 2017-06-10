#!/usr/bin/env python3
import sys
import os
import argparse
from collections import defaultdict
import time
import itertools
import theano
import theano.tensor as T
import numpy as np
import json
import Levenshtein
import colibricore


UNKFEATURE = -1
PUNCTFEATURE = -2


def compute_vector_distances(trainingdata, testdata):
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

    #lamblinsTrick = False

    #if lamblinsTrick:
    #    s = squaredPairwiseDistances
    #    bestIndices = T.cast( ( T.arange(n).dimshuffle(0, 'x') * T.cast(T.eq(s, s.min(axis=0, keepdims=True)), 'float32') ).sum(axis=0), 'int32')
    #else:
    #    bestIndices = T.argmin(squaredPairwiseDistances, axis=0)
    #nearests_fn = theano.function([x, y], bestIndices, profile=False)
    #return nearests_fn(trainingdata, testdata)

    squaredpwdist_fn = theano.function([x, y], T.transpose(squaredPairwiseDistances), profile=False)


    return squaredpwdist_fn(trainingdata, testdata)


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

def getfrequencytuple(candidate, patternmodel, lexicon, classencoder, lexfreq):
    """Returns a ( freq (int), inlexicon (bool) ) tuple"""
    pattern = classencoder.buildpattern(candidate)
    freq = patternmodel[pattern]
    if freq > 0:
        return freq, pattern in lexicon
    if pattern in lexicon:
        return lexfreq, True
    return 0, False

def trainingpatterns(patternmodel, lexicon, minfreq):
    for pattern in patternmodel:
        if len(pattern) == 1: #only unigrams for now
            if patternmodel[pattern] > minfreq:
                yield pattern
    for pattern in lexicon:
        if len(pattern) == 1: #only unigrams for now
            yield pattern

def timer(begintime):
    duration = time.time() - begintime
    print(" ^-- took " + str(round(duration,5)) + ' s', file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--patternmodel', type=str,help="Pattern model of a background corpus (training data; Colibri Core unindexed patternmodel)", action='store',required=True)
    parser.add_argument('-l','--lexicon', type=str,help="Lexicon file (training data; plain text, one word per line)", action='store',required=False)
    parser.add_argument('-c','--classfile', type=str,help="Class file of background corpus", action='store',required=True)
    parser.add_argument('-k','--neighbours','--neighbors', type=int,help="Maximum number of anagram neighbours to consider", action='store',default=20,required=False)
    parser.add_argument('-n','--topn', type=int,help="Maximum number of candidates to return", action='store',default=10,required=False)
    parser.add_argument('-D','--maxld', type=int,help="Maximum levenshtein distance", action='store',default=5,required=False)
    parser.add_argument('-t','--minfreq', type=int,help="Minimum frequency threshold (occurrence count) in background corpus", action='store',default=1,required=False)
    parser.add_argument('-a','--alphafreq', type=int,help="Minimum alphabet frequency threshold (occurrence count); characters occuring less are not considered in the anagram vectors", action='store',default=10,required=False)
    parser.add_argument('--lexfreq', type=int,help="Artificial frequency (occurrence count) for items in the lexicon that are not in the background corpus", action='store',default=1,required=False)
    parser.add_argument('--ldweight', type=float,help="Levenshtein distance weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--vdweight', type=float,help="Vector distance weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--freqweight', type=float,help="Frequency weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--lexweight', type=float,help="Lexicon distance weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--punctweight', type=int,help="Punctuation character weight for anagram vector representation", action='store',default=1,required=False)
    parser.add_argument('--unkweight', type=int,help="Unknown character weight for anagram vector representation", action='store',default=1,required=False)
    parser.add_argument('--json',action='store_true', help="Output JSON")
    parser.add_argument('-d', '--debug',action='store_true')
    args = parser.parse_args()

    if not args.lexicon:
        print("WARNING: You did not provide a lexicon! This will have a strong negative effect on the results!")
    elif not os.path.exists(args.lexicon):
        print("Error: Lexicon file " + args.lexicon + " does not exist",file=sys.stderr)
        sys.exit(2)

    if args.lexfreq < args.minfreq:
        print("WARNING: Lexicon base frequency is smaller than minimum frequency!",file=sys.stderr)
    if args.neighbours < args.topn:
        print("WARNING: Neighbour threshold (-k) is lower than return candidate threshold (-n)!",file=sys.stderr)


    if not os.path.exists(args.classfile):
        print("Error: Class file " + args.classfile + " does not exist",file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.patternmodel):
        print("Error: Pattern model file " + args.patternmodel + " does not exist",file=sys.stderr)
        sys.exit(2)

    print("Normalized weights used in candidate ranking:", file=sys.stderr)
    totalweight = args.ldweight + args.vdweight + args.freqweight + args.lexweight
    args.vdweight = args.vdweight / totalweight
    args.ldweight = args.ldweight / totalweight
    args.freqweight = args.freqweight / totalweight
    args.lexweight = args.lexweight / totalweight
    print(" Vector distance weight: ", args.vdweight , file=sys.stderr)
    print(" Levenshtein distance weight: ", args.ldweight, file=sys.stderr)
    print(" Frequency weight: ", args.freqweight, file=sys.stderr)
    print(" Lexicon weight: ", args.lexweight, file=sys.stderr)

    print("Test input words, one per line (if interactively invoked, type ctrl-D when done):",file=sys.stderr)
    testwords = [ w.strip() for w in sys.stdin.readlines() ]

    numtest = len(testwords)
    print("Test set size: ", numtest, file=sys.stderr)

    print("Loading background corpus from " + args.patternmodel, file=sys.stderr)
    begintime = time.time()
    classencoder= colibricore.ClassEncoder(args.classfile)
    patternmodel = colibricore.UnindexedPatternModel(args.patternmodel)
    timer(begintime)

    print("Loading lexicon... ", numtest, file=sys.stderr)
    lexicon = colibricore.UnindexedPatternModel()
    if args.lexicon:
        with open(args.lexicon,'r',encoding='utf-8') as f:
            for word in f:
                word = word.strip()
                if word:
                    pattern = classencoder.buildpattern(word, autoaddunknown=True) #adds lexicon words to the classencoder if they don't exist yet
                    lexicon.add(pattern)

        classencoder.save(args.classfile + '.extended')
        classdecoder = colibricore.ClassDecoder(args.classfile + '.extended')
    else:
        classdecoder = colibricore.ClassDecoder(args.classfile)
    alphabet = defaultdict(int)


    print("Computing alphabet on training data...",file=sys.stderr)
    begintime = time.time()
    for pattern in trainingpatterns(lexicon, patternmodel, args.minfreq):
        word = pattern.tostring(classdecoder) #string representation
        for char in word:
            if char.isalpha():
                alphabet[char] += 1
    timer(begintime)
    if args.debug: print("[DEBUG] Alphabet count: ", alphabet)


    #maps each character to a feature number (order number)
    alphabetmap = {}
    alphabetsize = 0
    for i, (char, freq) in enumerate(sorted(alphabet.items())):
        if freq >= args.alphafreq:
            alphabetmap[char] = alphabetsize
            alphabetsize += 1

    print("Alphabet computed (size=" + str(alphabetsize)+"): ", list(sorted(alphabetmap.keys())), file=sys.stderr)
    numfeatures = alphabetsize + 2 #UNK feature, PUNCT feature
    if args.debug: print(alphabetmap)

    print("Building test vectors...", file=sys.stderr)
    testdata = np.array( [ buildfeaturevector(testword, alphabetmap, numfeatures, args ) for testword in testwords] )
    if args.debug: print("[DEBUG] TEST DATA: ", testdata)


    numtraining = 0
    anahashcount = defaultdict(int) #frequency count of all seen anahashes (uses to determine which are actual anagrams in the training data)

    print("Counting anagrams in training data", file=sys.stderr)
    begintime = time.time()
    for pattern in trainingpatterns(lexicon, patternmodel, args.minfreq):
        word = pattern.tostring(classdecoder)
        h = anahash(word, alphabetmap, numfeatures)
        anahashcount[h] += 1
        if anahashcount[h] == 1:
            numtraining += 1
    timer(begintime)

    print("Training set size (anagram vectors): ", numtraining, file=sys.stderr)
    print("Background corpus size (patterns): ", len(patternmodel), file=sys.stderr)
    print("Lexicon size (patterns): ", len(lexicon), file=sys.stderr)

    print("Building training vectors and counting anagram hashes...", file=sys.stderr)
    begintime = time.time()
    trainingdata = np.empty((numtraining, numfeatures), dtype=np.int8)
    instanceindex = 0
    for pattern in trainingpatterns(lexicon, patternmodel, args.minfreq):
        word = pattern.tostring(classdecoder)
        h = anahash(word, alphabetmap, numfeatures)
        if anahashcount[h] >= 1:
            anahashcount[h] = anahashcount[h] * -1  #flip sign to indicate we visited this anagram already, prevent duplicates in training data
            trainingdata[instanceindex] = buildfeaturevector(word, alphabetmap, numfeatures, args)
            instanceindex  += 1
    timer(begintime)
    if args.debug: print("[DEBUG] TRAINING DATA DIMENSIONS: ", trainingdata.shape)

    print("Computing vector distances between test and trainingdata...", file=sys.stderr)
    begintime = time.time()
    distancematrix = compute_vector_distances(trainingdata, testdata)
    timer(begintime)

    print("Collecting matching anagrams", file=sys.stderr)
    begintime = time.time()
    matchinganagramhashes = defaultdict(set) #map of matching anagram hash to test words that yield it as a match
    for i, (testword, distances) in enumerate(zip(testwords, distancematrix)):
        #distances contains the distances between testword and all training instances
        #we extract the top k:
        for n, (distance, trainingindex) in enumerate(sorted(( (x,j) for j,x in enumerate(distances) ))): #MAYBE TODO: delegate to numpy/theano if too slow?
            if n == args.neighbours:
                break
            h = anahash_fromvector(trainingdata[trainingindex])
            matchinganagramhashes[h].add((testword, distance))
    timer(begintime)

    print("Resolving anagram hashes to candidates", file=sys.stderr)
    begintime = time.time()
    candidates = defaultdict(list) #maps test words to  candidates (str => [str])
    for pattern in trainingpatterns(lexicon, patternmodel, args.minfreq):
        trainingword = pattern.tostring(classdecoder)
        h = anahash(trainingword, alphabetmap, numfeatures)
        if h in matchinganagramhashes:
            for testword, vectordistance in matchinganagramhashes[h]:
                candidates[testword].append((trainingword,vectordistance))
    timer(begintime)

    print("Ranking candidates...", file=sys.stderr)
    begintime = time.time()
    results = []
    #output in same order as input
    for testword in testwords:
        #we have multiple candidates per testword; we are going to use four sources to rank:
        #   1) the vector distance
        #   2) the levensthein distance
        #   3) the frequency in the background corpus
        #   4) the presence in lexicon or not
        candidates_extended = [ (candidate, vdistance, Levenshtein.distance(testword, candidate), getfrequencytuple(candidate, patternmodel, lexicon, classencoder, args.lexfreq)) for candidate, vdistance in candidates[testword] ]
        #prune candidates below thresholds:
        candidates_extended = [ (candidate, vdistance, ldistance, freqtuple[0], freqtuple[1]) for candidate, vdistance, ldistance,freqtuple in candidates_extended if ldistance <= args.maxld and freqtuple[0] >= args.minfreq ]
        result_candidates = []
        if candidates_extended:
            freqsum = sum(( freq for _, _, _, freq, _  in candidates_extended ))

            #compute a normalized compound score including all components according to their weights:
            candidates_scored = [ ( candidate, (
                args.vdweight * (1/(vdistance+1)) + \
                args.ldweight * (1/(ldistance+1)) + \
                args.freqweight * (freq/freqsum) + \
                (args.lexweight if inlexicon else 0)
                )
            ,vdistance, ldistance, freq, inlexicon) for candidate, vdistance, ldistance, freq, inlexicon in candidates_extended ]

            #output candidates:
            for i, (candidate, score, vdistance, ldistance,freq, inlexicon) in enumerate(sorted(candidates_scored, key=lambda x: -1 * x[1])):
                if i == args.neighbours: break
                result_candidates.append( {'text': candidate,'score': score, 'vdistance': vdistance, 'ldistance': ldistance, 'freq': freq, 'inlexicon': inlexicon } )

        result = {'text': testword, 'candidates': result_candidates}
        results.append(result)
    timer(begintime)

    if args.json:
        print("Outputting JSON...", file=sys.stderr)
        print(json.dumps(results))
    else:
        print("Outputting Text (use --json for full output in JSON)...", file=sys.stderr)
        for result in results:
            print(result['text'])
            for candidate in result['candidates']:
                print("\t" + candidate['text'] + "\t[score=" + str(candidate['score']) + " vd=" + str(candidate['vdistance']) + " ld=" + str(candidate['ldistance']) + " freq=" + str(candidate['freq']) + " inlexicon=" + str(int(candidate['inlexicon'])) + "]")


    return results

if __name__ == '__main__':
    main()
