#!/usr/bin/env python3

#-------------------------------------------------------------------
# Anavec Spelling Correction and Normalisation System
#-------------------------------------------------------------------
#       by Maarten van Gompel
#       Radboud University Nijmegen
#       Licensed under GPLv3

import sys
import os
import argparse
from collections import defaultdict
import time
import itertools
import math
import theano
import theano.tensor as T
import numpy as np
import json
import Levenshtein
import colibricore
from pynlpl.datatypes import PriorityQueue
from pynlpl.algorithms import possiblesplits
try:
    import kenlm
    HASLM= True
except:
    print("WARNING: KenLM is not installed, LM support not available. See https://github.com/kpu/kenlm", file=sys.stderr)
    HASLM = False

UNKFEATURE = -1
PUNCTFEATURE = -2

class InputTokenState:
    CORRECTABLE = 0 #The input word is either correct or incorrect, it is up to the system to determine and correct it (default state)
    CORRECT = 1 #The input word is correct, nothing is to be done for it (it is only used as context)
    INCORRECT = 2 #The input word is explicitly incorrect, it has to be corrected


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





def anahash_fromvector(vector):
    hashvalue = 0
    for i, count in enumerate(vector):
        hashvalue += count * ((100+i)**5)
    return hashvalue


def timer(begintime):
    duration = time.time() - begintime
    print(" ^-- took " + str(round(duration,5)) + ' s', file=sys.stderr)

def ngrams(seq, n):
    """Yields an n-gram (tuple) at each iteration"""
    l = len(seq)

    for i in range(-(n - 1),l):
        begin = i
        end = i + n
        if begin >= 0 and end <= l:
            ngram = seq[begin:end]
            yield ngram

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def combinations(l):
    #[('to', 'too'), ('be', 'bee'), ('happy', 'hapy', 'heppie')] -> [['to', 'be', 'happy'], ... ]
    if not l:
        yield []
    head = l[0]
    tail = l[1:]
    if not tail: #stop condition
        for x in head:
            yield [x]
    else: #recursion step
        for x in head:
            for y in combinations(tail):
                yield [x] + y



class Corrector:
    def __init__(self, *testwords, **args):
        self.args = AttributeDict(args)

        if not self.args.lexicon:
            print("WARNING: You did not provide a lexicon! This will have a strong negative effect on the results!")
        elif not os.path.exists(self.args.lexicon):
            print("Error: Lexicon file " + self.args.lexicon + " does not exist",file=sys.stderr)
            sys.exit(2)

        if self.args.lexfreq < self.args.minfreq:
            print("WARNING: Lexicon base frequency is smaller than minimum frequency!",file=sys.stderr)


        if not os.path.exists(self.args.classfile):
            print("Error: Class file " + self.args.classfile + " does not exist",file=sys.stderr)
            sys.exit(2)
        if not os.path.exists(self.args.patternmodel):
            print("Error: Pattern model file " + self.args.patternmodel + " does not exist",file=sys.stderr)
            sys.exit(2)

        print("Normalized weights used in candidate ranking:", file=sys.stderr)
        totalweight = self.args.ldweight + self.args.vdweight + self.args.freqweight + self.args.lexweight
        self.args.vdweight = self.args.vdweight / totalweight
        self.args.ldweight = self.args.ldweight / totalweight
        self.args.freqweight = self.args.freqweight / totalweight
        self.args.lexweight = self.args.lexweight / totalweight
        print(" Vector distance weight: ", self.args.vdweight , file=sys.stderr)
        print(" Levenshtein distance weight: ", self.args.ldweight, file=sys.stderr)
        print(" Frequency weight: ", self.args.freqweight, file=sys.stderr)
        print(" Lexicon weight: ", self.args.lexweight, file=sys.stderr)
        print("Normalized weights used in language model ranking:", file=sys.stderr)
        totalweight = self.args.lmweight + self.args.correctionweight
        self.args.lmweight = self.args.lmweight / totalweight
        self.args.correctionweight = self.args.correctionweight / totalweight
        print(" Language model weight: ", self.args.lmweight , file=sys.stderr)
        print(" Correction model weight: ", self.args.correctionweight, file=sys.stderr)


        print("Loading background corpus from " + self.args.patternmodel, file=sys.stderr)
        begintime = time.time()
        self.classencoder = colibricore.ClassEncoder(self.args.classfile)
        self.patternmodel = colibricore.UnindexedPatternModel(self.args.patternmodel) #background corpus
        timer(begintime)

        print("Loading lexicon... ", file=sys.stderr)
        self.lexicon = colibricore.UnindexedPatternModel()
        if self.args.lexicon:
            with open(self.args.lexicon,'r',encoding='utf-8') as f:
                for word in f:
                    word = word.strip()
                    if word:
                        pattern = self.classencoder.buildpattern(word, autoaddunknown=True) #adds lexicon words to the classencoder if they don't exist yet
                        self.lexicon.add(pattern)

            self.classencoder.save(self.args.classfile + '.extended')
            self.classdecoder = colibricore.ClassDecoder(self.args.classfile + '.extended')
        else:
            self.classdecoder = colibricore.ClassDecoder(self.args.classfile)
        alphabet = defaultdict(int)

        if self.args.lm:
            print("Loading language model... ", file=sys.stderr)
            if not HASLM:
                raise Exception("KenLM is not installed! Language Model support unavailable")
            self.lm = kenlm.Model(self.args.lm)
        else:
            self.lm = None

        print("Computing alphabet on training data...",file=sys.stderr)
        begintime = time.time()
        for pattern in self.trainingpatterns():
            word = pattern.tostring(self.classdecoder) #string representation
            for char in word:
                if char.isalpha():
                    alphabet[char] += 1
        timer(begintime)
        if self.args.debug: print("[DEBUG] Alphabet count: ", alphabet)


        #maps each character to a feature number (order number)
        self.alphabetmap = {}
        if self.args.ngrams > 1:
            self.alphabetmap[' '] = 1
            alphabetsize = 1
        else:
            alphabetsize = 0
        for i, (char, freq) in enumerate(sorted(alphabet.items())):
            if freq >= self.args.alphafreq:
                self.alphabetmap[char] = alphabetsize
                alphabetsize += 1

        print("Alphabet computed (size=" + str(alphabetsize)+"): ", list(sorted(self.alphabetmap.keys())), file=sys.stderr)
        self.numfeatures = alphabetsize + 2 #UNK feature, PUNCT feature
        if self.args.debug: print(self.alphabetmap)



        self.numtraining = 0
        anahashcount = defaultdict(int) #frequency count of all seen anahashes (uses to determine which are actual anagrams in the training data)

        print("Counting anagrams in training data", file=sys.stderr)
        begintime = time.time()
        for pattern in self.trainingpatterns():
            word = pattern.tostring(self.classdecoder)
            h = self.anahash(word)
            anahashcount[h] += 1
            if anahashcount[h] == 1:
                self.numtraining += 1
        timer(begintime)

        print("Training set size (anagram vectors): ", self.numtraining, file=sys.stderr)
        print("Background corpus size (patterns): ", len(self.patternmodel), file=sys.stderr)
        print("Lexicon size (patterns): ", len(self.lexicon), file=sys.stderr)

        print("Building training vectors and counting anagram hashes...", file=sys.stderr)
        begintime = time.time()
        self.trainingdata = np.empty((self.numtraining, self.numfeatures), dtype=np.int8)
        instanceindex = 0
        for pattern in self.trainingpatterns():
            word = pattern.tostring(self.classdecoder)
            h = self.anahash(word)
            if anahashcount[h] >= 1:
                anahashcount[h] = anahashcount[h] * -1  #flip sign to indicate we visited this anagram already, prevent duplicates in training data
                self.trainingdata[instanceindex] = self.buildfeaturevector(word)
                instanceindex  += 1
        timer(begintime)
        if self.args.debug: print("[DEBUG] TRAINING DATA DIMENSIONS: ", self.trainingdata.shape)

    def correct(self, testtokens, mask=None):
        """Correct the testtokens (a list of strings in a pseudo-tokenised representation), an empty element or "\n" element is allowed as an explicit sentence seperator.

        The optional mask parameter is an equally-sized sequence, corresponding to the testwords, that
        fine-tunes which words to correct, indicated by the following values:
            0 - InputTokenState.CORRECTABLE  -- The input word is either correct or incorrect, it is up to the system to determine and correct it (default state)
            1 - InputTokenState.CORRECT  #The input word is correct, nothing is to be done for it (it is only used as context)
            2 - InputTokenState.INCORRECT  #The input word is explicitly incorrect, it has to be corrected
        """

        if not mask:
            mask = [InputTokenState.CORRECTABLE for word in testtokens]

        if len(mask) != len(testtokens):
            raise Exception("Supplied mask must be as long as the testtokens!")

        testpatterns = list(self.gettestpatterns(testtokens, mask))

        numtest = len(testpatterns)
        print("Test set model size: ", numtest, file=sys.stderr)

        print("Building test vectors...", file=sys.stderr)
        testdata = np.array( [ self.buildfeaturevector(testword) for testword, _,_,_ in testpatterns] )
        if self.args.debug: print("[DEBUG] TEST DATA: ", testdata)

        print("Computing vector distances between test and trainingdata...", file=sys.stderr)
        begintime = time.time()
        distancematrix = compute_vector_distances(self.trainingdata, testdata)
        timer(begintime)

        print("Collecting matching anagrams", file=sys.stderr)
        begintime = time.time()
        matchinganagramhashes = defaultdict(set) #map of matching anagram hash to test words that yield it as a match
        testinstance = 0
        getdistancematrix = iter(distancematrix)
        for (testword, state, index, length), distances in zip(testpatterns,distancematrix):
            #- testword is the actual text (str) of the pattern, usually a word
            #- index is the index in the original testwords
            #- length is the length of the pattern (in tokens)

            #distances contains the distances between testword and all training instances
            #we extract the top k:
            matchingdistances = set()
            for distance, trainingindex in sorted(( (x,j) for j,x in enumerate(distances) )): #MAYBE TODO: delegate to numpy/theano if too slow?
                matchingdistances.add(distance)
                if len(matchingdistances) > self.args.neighbours:
                    break
                h = anahash_fromvector(self.trainingdata[trainingindex])
                matchinganagramhashes[h].add((testword, distance))
        timer(begintime)

        print("Resolving anagram hashes to candidates", file=sys.stderr)
        begintime = time.time()
        candidates = defaultdict(list) #maps test words to  candidates (str => [str])
        for pattern in self.trainingpatterns():
            trainingword = pattern.tostring(self.classdecoder)
            h = self.anahash(trainingword)
            if h in matchinganagramhashes:
                for testword, vectordistance in matchinganagramhashes[h]:
                    candidates[testword].append((trainingword,vectordistance))
        timer(begintime)

        print("Scoring and ranking candidates...", file=sys.stderr)
        begintime = time.time()

        #We collect all candidates in a big candidate tree, where a list of candidates is stored for each test index, and each possible token length
        candidatetree = {} #index number => length => [candidates]

        for testword, state, index, length in testpatterns:
            if not index in candidatetree: candidatetree[index] = defaultdict(list)
            if state == InputTokenState.CORRECT:
                #Word is already correct and not to be tested, just look up some data and copy to output
                freqtuple = self.getfrequencytuple(testword)
                assert length == 1
                candidatetree[index][length].append(
                    AttributeDict({'text':testword,'score': 1.0, 'ldistance': 0, 'vdistance': 0.0, 'freq': freqtuple[0], 'inlexicon': freqtuple[1], 'correct':1, 'lmchoice':False})
                )
            else:
                if self.args.debug: print("[DEBUG] Candidates for '" + testword + "' prior to pruning: " + str(len(candidates[testword])),file=sys.stderr)
                #we have multiple candidates per testword; we are going to use four sources to rank:
                #   1) the vector distance
                #   2) the levensthein distance
                #   3) the frequency in the background corpus
                #   4) the presence in lexicon or not
                candidates_extended = [ (candidate, vdistance, Levenshtein.distance(testword, candidate), self.getfrequencytuple(candidate)) for candidate, vdistance in candidates[testword] ]
                #prune candidates below thresholds:
                candidates_extended = [ (candidate, vdistance, ldistance, freqtuple[0], freqtuple[1]) for candidate, vdistance, ldistance,freqtuple in candidates_extended if ldistance <= self.args.maxld and freqtuple[0] >= self.args.minfreq ]
                if self.args.debug: print("[DEBUG] Candidates for '" + testword + "' after frequency & LD pruning (" + str(len(candidates_extended))+ ")", candidates_extended,file=sys.stderr)
                if state == InputTokenState.INCORRECT:
                    #The word is explicitly marked incorrect, any correction candidate that is equal to the input word will be pruned
                    candidates_extended = [ (candidate, vdistance, ldistance, freq, inlexicon) for candidate, vdistance, ldistance,freq, inlexicon  in candidates_extended if candidate != testword ]
                result_candidates = []
                if candidates_extended:
                    freqsum = sum(( freq for _, _, _, freq, _  in candidates_extended ))

                    #compute a confidence score including all components according to their weights:
                    candidates_confidencescored = [ ( candidate, (
                        self.args.vdweight * (1/(vdistance+1)) + \
                        self.args.ldweight * (1/(ldistance+1)) + \
                        self.args.freqweight * (freq/self.patternmodel.tokens()) + \
                        (self.args.lexweight if inlexicon else 0)
                        )
                    ,vdistance, ldistance, freq, inlexicon) for candidate, vdistance, ldistance, freq, inlexicon in candidates_extended ]

                    #collect candidates in candidate tree
                    #we transform the confidence score into a likelihood P(correction|original) by normalising over all candidates and taking into account their rank number
                    candidates_confidencescored.sort(key=lambda x: -1*x[1]) #sort based on confidence score, descending
                    candidates_confidencescored = candidates_confidencescored[:self.args.candidates] #prune candidates below the cut-off threshold
                    confidencesum = sum( ( score for _,score,_,_,_,_ in candidates_confidencescored) )
                    for i, (candidate, score, vdistance, ldistance,freq, inlexicon) in enumerate(sorted(candidates_confidencescored, key=lambda x: -1 * x[1])):
                        logprob = math.log10(score)
                        candidatetree[index][length].append(
                            AttributeDict({'text': candidate,'logprob': logprob, 'score': score, 'vdistance': vdistance, 'ldistance': ldistance, 'freq': freq, 'inlexicon': inlexicon, 'correct': i == 0 and candidate == testword and score >= self.args.correctscore, 'lmchoice': False})
                        )
        timer(begintime)

        if self.args.simple:
            #Simple approach
            self.applylm(candidatetree, testtokens)
            results = [ {'text': w, 'candidates': candidatetree[i][1] } for i, w in enumerate(testtokens) ]
        else:
            #Full context-aware approach
            print("Decoding...", file=sys.stderr)
            decoder = StackDecoder(self, testtokens, mask, candidatetree, self.args.beamsize)
            topresults = []
            results = {'top':topresults, 'candidatetree': candidatetree, 'testtokens': testtokens, 'mask': mask  } #contains the top n best results
            for i, hyp in enumerate(decoder.decode(self.args.topn)):
                topresults.append(hyp)
        return results

    def applylm(self, candidatetree, testtokens):
        """Applies the Language Model to favour certain candidates in the simple mode (superseded by full beam decoder)"""
        #we run words that are correctable (i.e. not marked correct), through a language model, using all candidate permutations (above a certain cut-off)
        print("Applying Language Model...", file=sys.stderr)

        #first we identify sequences of correctable words to pass to the language model along with some correct context (the first is always a correct token (or begin of sentence), and the last a correct token (or end of sentence))
        leftcontext = []
        i = 0
        while i < len(testtokens):
            if i in candidatetree and 1 in candidatetree[i]:
                if candidatetree[i][1][0].correct:
                    leftcontext.append(testtokens[i])
                else:
                    #we found a correctable word
                    span = 1 #span of correctable words in tokens/words
                    rightcontext = []
                    j = i+1
                    while j < len(testtokens):
                        if j in candidatetree and 1 in candidatetree[j] and candidatetree[j][1][0].correct:
                            rightcontext.append(testtokens[j])
                        elif rightcontext:
                            break
                        else:
                            span += 1
                        j += 1

                    #[('to', 'too'), ('be', 'bee'), ('happy', 'hapy')] (with with dicts instead of strings)
                    allcandidates = [ candidatetree[j][1] for j in range(i,i+span) ]

                    allcombinations = list(combinations(allcandidates))
                    if self.args.debug: print("[DEBUG LM] Examining " + str(len(allcombinations)) + " possible combinations for '" + " ".join(testtokens[i:i+span]) + "'",file=sys.stderr)

                    bestlmscore = 0
                    bestspanscore = 0 # best span score

                    scores = []

                    #obtain LM scores for all combinations
                    for spancandidates in allcombinations:
                        text = " ".join(leftcontext + [ candidate.text for candidate in spancandidates] + rightcontext)
                        lmscore = 10 ** self.lm.score(text, bos=(len(leftcontext)>0), eos=(len(rightcontext)>0))  #kenlm returns logprob
                        if lmscore >= bestlmscore:
                            bestlmscore = lmscore
                        spanscore = np.prod([candidate.score for candidate in spancandidates])
                        if spanscore > bestspanscore:
                            bestspanscore = spanscore
                        scores.append((lmscore, spanscore))

                    #Compute a normalized span score that includes the correction scores of the individual candidates as well as the over-arching LM score
                    bestcombination = None
                    besttotalscore = 0
                    for spancandidates, (lmscore, spanscore) in zip(allcombinations, scores):
                        totalscore = self.args.lmweight * (lmscore/bestlmscore) + self.args.correctionweight * (spanscore/bestspanscore)
                        if self.args.debug: print("[DEBUG LM] text=" + " ".join(leftcontext + [ candidate.text for candidate in spancandidates] + rightcontext) + " totalscore=" + str(totalscore) + " lmscore=" + str(lmscore/bestlmscore) + " spanscore=" + str(spanscore/bestspanscore) + " leftcontext=" + " ".join(leftcontext) + " rightcontext=" + " ".join(rightcontext),file=sys.stderr)
                        if totalscore > besttotalscore:
                            besttotalscore = totalscore
                            bestcombination = spancandidates

                    #the best combination gets selected by the LM
                    for candidate in bestcombination:
                        candidate.lmchoice = True

                    if self.args.debug: print("[DEBUG LM] Language model selected " + " ".join([candidate.text for candidate in bestcombination]) + " with a total score of ", besttotalscore,file=sys.stderr)

                    #reorder the candidates in the output so that the chosen candidate is always the first
                    #for result in results[i:i+span]:
                    #    lmchoice = 0
                    #    for j, candidate in enumerate(result.candidates):
                    #        if j == 0:
                    #            if candidate.lmchoice:
                    #                break #nothing to do
                    #        elif candidate.lmchoice:
                    #            lmchoice = j
                    #    if lmchoice != 0:
                    #        result.candidates = [result.candidates[lmchoice]] + result.candidates[:lmchoice]  + result.candidates[lmchoice+1:]

                    leftcontext = leftcontext + [candidate.text for candidate in bestcombination] + rightcontext
                    if len(leftcontext) > 15: leftcontext = leftcontext[-15:]  #don't let it grow too big
                    i = i + span + len(rightcontext) - 1
            i += 1





    def output_json(self, results):
        print("Outputting JSON...", file=sys.stderr)
        print(json.dumps(results, indent=4, ensure_ascii=False))

    def output_simple(self, results):
        print("Outputting Text (use --json for full output in JSON)...", file=sys.stderr)
        for result in results:
            print(result['text'])
            for candidate in result['candidates']:
                print("\t" + candidate['text'] + "\t[score=" + str(candidate['score']) + " vd=" + str(candidate['vdistance']) + " ld=" + str(candidate['ldistance']) + " freq=" + str(candidate['freq']) + " inlexicon=" + str(int(candidate['inlexicon'])) + " correct=" + str(int(candidate['correct'])),end="")
                if 'lmchoice' in candidate:
                    print(" lmchoice=" + str(int(candidate.lmchoice)), end="")
                print("]")

    def output(self, results):
        print("Outputting best text (use --json for full output in JSON or --report for a report in text)...", file=sys.stderr)
        if self.args.simple:
            print(" ".join([w.text for w in results]))
        else:
            print(str(results['top'][0]))

    def output_report(self, results):
        print("Outputting report (use --json for full parsable output in JSON)...", file=sys.stderr)
        for i, result in enumerate(results['top']):
            print("RESULT #" + str(i+1) + ": " + str(result))
            print("  Representation: " + repr(result))
        print("CANDIDATE TREE:")
        for index in results['candidatetree']:
            for length in results['candidatetree'][index]:
                print("@" + str(index) + ":" + str(length) + " " + " ".join(results['testtokens'][index:index+length]))
                for candidate in results['candidatetree'][index][length]:
                    print("\t" + candidate['text'] + "\t[score=" + str(candidate['score']) + " logprob="+str(candidate.logprob) + " vd=" + str(candidate['vdistance']) + " ld=" + str(candidate['ldistance']) + " freq=" + str(candidate['freq']) + " inlexicon=" + str(int(candidate['inlexicon'])) + " correct=" + str(int(candidate['correct'])) + "]")

    def getfrequencytuple(self, candidate):
        """Returns a ( freq (int), inlexicon (bool) ) tuple"""
        pattern = self.classencoder.buildpattern(candidate)
        try:
            freq = self.patternmodel[pattern]
        except KeyError:
            freq = 0
        if freq > 0:
            if len(pattern) == 1:
                inlexicon = pattern in self.lexicon
            else:
                inlexicon = all((unigram in self.lexicon for unigram in pattern.ngrams(1)))
            return freq, inlexicon
        if pattern in self.lexicon:
            return self.args.lexfreq, True
        return 0, False

    def trainingpatterns(self):
        """Iterate over all patterns in the training data (background corpus + lexicon)"""
        for pattern in self.patternmodel:
            if (self.args.simple and len(pattern) == 1)  or (not self.args.simple and len(pattern) <= self.args.ngrams):  #in simple mode only unigrams, otherwise all
                if self.patternmodel[pattern] > self.args.minfreq:
                    yield pattern
        for pattern in self.lexicon:
            if ((self.args.simple and len(pattern) == 1)  or (not self.args.simple and len(pattern) <= self.args.ngrams)) and pattern not in self.patternmodel: #last condition prevents yielding duplicates
                yield pattern

    def gettestpatterns(self, testtokens, mask):
        testpatterns = []
        if self.args.ngrams >= 1:
            sequences = []
            sequence = []

        for i, (text, state) in enumerate(zip(testtokens, mask)):
            if state != InputTokenState.CORRECT and text.strip():
                if self.args.ngrams >= 1: sequence.append( (text, state,i ) )
                testpatterns.append( (text, state, i, 1) )
            else:
                if self.args.ngrams >= 1 and sequence:
                    sequences.append(sequence)
                    sequence = []

        if self.args.ngrams >= 1 and not self.args.simple:
            if sequence: sequences.append(sequence) #sequences contains a sequence of consecutive correctable tokens
            if sequences:
                for n in range(2,self.args.ngrams+1):
                    for sequence in sequences:
                        for ngram in ngrams(sequence, n):
                            text = "".join( w for w, _,_ in ngram )
                            index = min( i for _,_,i in ngram)
                            state = max( s for _,s,_ in ngram) #will be CORRECTABLE or INCORRECT ( CORRECT can't be part of a sequence)
                            testpatterns.append(( text, state, index, n ))

        return testpatterns


    def buildfeaturevector(self, word):
        featurevector = np.zeros(self.numfeatures, dtype=np.uint8)
        for char in word:
            if not char.isalnum():
                featurevector[PUNCTFEATURE] += 1 * self.args.punctweight
            elif char in self.alphabetmap:
                featurevector[self.alphabetmap[char]] += 1
            else:
                featurevector[UNKFEATURE] += 1 * self.args.unkweight
        return featurevector

    def anahash(self, word):
        hashvalue = 0
        for char in word:
            if not char.isalnum():
                charvalue = 100 + self.numfeatures + PUNCTFEATURE
            elif char in self.alphabetmap:
                charvalue = 100 + self.alphabetmap[char]
            else:
                charvalue = 100 + self.numfeatures + UNKFEATURE
            hashvalue += charvalue**5
        return hashvalue

class StackDecoder:
    def __init__(self, corrector, testwords, mask, candidatetree, beamsize):
        self.corrector = corrector
        self.testwords = testwords
        self.mask = mask
        self.candidatetree = candidatetree
        self.length = len(self.testwords)
        self.beamsize = beamsize
        self.computemaxprob()
        self.stacks = []
        for i in range(0,self.length+1):
            self.stacks.append(PriorityQueue([], lambda x: x.logprob, length=self.beamsize)) #minimize because we will be working with logprobs (base 10)
        #add initial hypothesis:
        self.stacks[0].append(CorrectionHypothesis(None,0,0,self))

    def decode(self, topn=1):
        for i, stack in enumerate(self):
            if i < self.length:
                print("Decoding stack " + str(i) + "...", file=sys.stderr)
                count = 0
                while stack.data:
                    hypothesis = stack.pop()
                    for newhypothesis in hypothesis.expand():
                        count += 1
                        self.stacks[newhypothesis.coverage()].append(newhypothesis)
                print(" (" + str(count) + " hypotheses generated after decoding stack " + str(i) + ")", file=sys.stderr)
                if self.corrector.args.debug: print("[DEBUG] Stack sizes after decoding stack " + str(i)+ ": ", [ (i, len(s)) for i, s in enumerate(self.stacks) ])
        for i in range(0,topn):
            if self.stacks[self.length].data:
                yield self.stacks[self.length].pop() #return the best (=first) hypothesis in the last stack

    def __iter__(self):
        for stack in self.stacks:
            yield stack

    def __len__(self):
        return self.length

    def __getitem__(self, stackindex):
        return self.stacks[stackindex]

    def append(self, stackindex, hypothesis):
        self.stacks[stackindex].append(hypothesis)

    def computemaxprob(self):
        """precompute and store a maximal score (minimal cost) for all possible contiguous sequences, this will be used in future cost estimation in the beam search"""

        print("Precomputing maximum probability (minimal cost) paths for decoder...",file=sys.stderr)

        #note that in this function we minimize rather than maximize as we are useing logprobs everywhere

        splits = {}
        for n in range(2, self.length+1):
            splits[n] = list(possiblesplits(n))

        #populate the maxprob matrix will costs directly from the candidates
        self.maxprob = {}
        for index in range(0, self.length):
            for length in range(1, self.length-index+1):
                try:
                    candidates = self.candidatetree[index][length]
                except KeyError:
                    candidates = None
                p = -9999
                if candidates:
                    if self.corrector.lm:
                        p = max( ( self.corrector.args.correctionweight * candidate.logprob + self.corrector.args.lmweight * self.corrector.lm.score(candidate.text) for candidate in candidates ) )
                    else:
                        p = max( ( self.corrector.args.correctionweight * candidate.logprob for candidate in candidates ) )
                self.maxprob[(index, length)] = p

        #now ensure the score of any slice is not smaller than the maximal sum amongst its parts
        #also assign cost to previously uncovered slices
        for index in range(0, self.length):
            for length in range(2, self.length-index):
                if (index, length) in self.maxprob.keys():
                    p = self.maxprob[(index,length)]
                else:
                    p = None #we have no cost for this cell yet
                for partition in splits[length]:
                    partitionsum = 0
                    for subindex, sublength in partition:
                        subindex = index + subindex
                        try:
                            partitionsum += self.maxprob[(subindex, sublength)]
                        except KeyError:
                            partitionsum = None
                            break
                    if partitionsum is not None and p < partitionsum:
                        p = partitionsum #assigns the minimal partitionsum
                if p is None or p < partitionsum:
                    self.maxprob[(index, length)] = partitionsum

        print("Size of max prob matrix:",len(self.maxprob), file=sys.stderr)
        if self.corrector.args.debug:
            print("[DEBUG] Max prob matrix:",self.maxprob, file=sys.stderr)



class CorrectionHypothesis:
    def __init__(self, candidate, index, length, decoder, parent=None):
        self.decoder = decoder #the decoder provides the necessary context, it is in turn tied to the corrector
        self.parent = parent #links to the parent hypothesis that generated this one
        self.candidate = candidate #or None for the initial root hypothesis
        self.index = index #the position of the last added candidate in the original testtokens sequence
        self.length = length #the length of the last added candidate
        #if parent is None:
        #    self.covered = np.zeros(len(self.decoder), dtype=np.byte)
        #else:
        #    self.covered = self.parent.covered.copy()
        #    self.covered[self.index:self.index+self.length+1] = 1
        self.logprob = self.computeprob()
        if self.decoder.corrector.args.debug:
            print("[DEBUG] Generated Hypothesis " + repr(self), file=sys.stderr)

    def expand(self):
        nextindex = self.index + self.length
        if nextindex < self.decoder.length:
            for length, candidates in self.decoder.candidatetree[nextindex].items():
                for candidate in candidates:
                    yield CorrectionHypothesis(candidate, nextindex, length, self.decoder, self)
                    if candidate.correct:
                        break #do not consider further alternatives if candidate is resolved as correct


    def __repr__(self):
        return "<"  + str(self) + " [logprob=" + str(self.logprob) + "; coverage=" + repr(self.coverage()) + "; correctionprob=" + str(self.correctionprob) + "; lmprob=" + str(self.lmprob) + "]>"

    def __str__(self):
        if self.candidate is None:
            return ""
        elif self.parent is None:
            return self.candidate.text
        else:
            return (str(self.parent) + " " + self.candidate.text).strip()

    def __lt__(self, other):
        return self.logprob < other.logprob

    def __gt__(self, other):
        return self.logprob > other.logprob

    def computeprob(self):
        """Returns the cost thus-far plus the minimum future cost of all uncovered parts, based on precomputed data"""

        #correction probability of the whole chain
        if self.parent is None:
            if self.candidate is None:
                self.correctionprob = 0
            else:
                self.correctionprob = self.candidate.logprob
        else:
            self.correctionprob = self.parent.correctionprob + self.candidate.logprob

        #cost of the hypothesis thus far
        if self.decoder.corrector.lm:
            self.lmprob = self.decoder.corrector.lm.score(str(self)) #base 10 logprob
            self.prob = (self.decoder.corrector.args.correctionweight * self.correctionprob) + (self.decoder.corrector.args.lmweight * self.lmprob)
        else:
            self.lmprob = 0
            self.prob = self.correctionprob


        self.futureprob = 0 #will be a logprob (base 10)
        #begin = None
        #length = 1
        #retrieve future cost for all consecutive uncovered parts
        if self.index + self.length < self.decoder.length:
            self.futureprob += self.decoder.maxprob[(self.index+self.length, self.decoder.length - (self.index+self.length))]
        #for i, indexcovered in enumerate(self.covered):
        #    if not indexcovered:
        #        #position is uncovered
        #        if begin is None:
        #            begin = indexcovered
        #            length = 1
        #        else:
        #            length += 1
        #    else:
        #        #position is covered
        #        if begin is not None:
        #           #process the last uncovered sequence
        #           self.futurecost += self.decoder.maxprob[(begin,length)]
        #           begin = None

        #if begin is not None: #do not forget to process any final uncovered sequence
        #    self.futurecost += self.decoder.maxprob[(begin,length)]

        return self.prob + self.futureprob

    def complete(self):
        return self.index + self.length == self.decoder.length
        #return np.all(self.covered)

    def coverage(self):
        return self.index + self.length
        #return np.sum(self.covered)







def setup_argparser(parser):
    parser.add_argument('-m','--patternmodel', type=str,help="Pattern model of a background corpus (training data; Colibri Core unindexed patternmodel)", action='store',required=True)
    parser.add_argument('-l','--lexicon', type=str,help="Lexicon file (training data; plain text, one word per line)", action='store',required=False)
    parser.add_argument('-L','--lm', type=str,help="Language model file in ARPA format", action='store',required=False)
    parser.add_argument('-c','--classfile', type=str,help="Class file of background corpus", action='store',required=True)
    parser.add_argument('-k','--neighbours','--neighbors', type=int,help="Maximum number of anagram distances to consider (the actual amount of anagrams is likely higher)", action='store',default=3, required=False)
    parser.add_argument('-K','--candidates', type=int,help="Maximum number of candidates  to consider per input token/pattern", action='store',default=100, required=False)
    parser.add_argument('-n','--topn', type=int,help="Maximum number of candidates to return", action='store',default=10,required=False)
    parser.add_argument('-N','--ngrams', type=int,help="N-grams to consider (max value of n). Ensure that your background corpus is trained for at least the same length for this to have any effect!", action='store',default=3,required=False)
    parser.add_argument('-D','--maxld', type=int,help="Maximum levenshtein distance", action='store',default=5,required=False)
    parser.add_argument('-t','--minfreq', type=int,help="Minimum frequency threshold (occurrence count) in background corpus", action='store',default=1,required=False)
    parser.add_argument('-a','--alphafreq', type=int,help="Minimum alphabet frequency threshold (occurrence count); characters occuring less are not considered in the anagram vectors", action='store',default=10,required=False)
    parser.add_argument('-b','--beamsize', type=int,help="Beamsize for the decoder", action='store',default=100,required=False)
    parser.add_argument('--lexfreq', type=int,help="Artificial frequency (occurrence count) for items in the lexicon that are not in the background corpus", action='store',default=1,required=False)
    parser.add_argument('--ldweight', type=float,help="Levenshtein distance weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--vdweight', type=float,help="Vector distance weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--freqweight', type=float,help="Frequency weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--lexweight', type=float,help="Lexicon distance weight for candidating ranking", action='store',default=1,required=False)
    parser.add_argument('--lmweight', type=float,help="Language Model weight for Language Model selection (together with --correctionweight)", action='store',default=1,required=False)
    parser.add_argument('--correctionweight', type=float,help="Correction Model weight for Language Model selection (together with --lmweight)", action='store',default=1,required=False)
    parser.add_argument('--correctscore', type=float,help="The score the a word must reach to be considered correct", action='store',default=0.60,required=False)
    parser.add_argument('--punctweight', type=int,help="Punctuation character weight for anagram vector representation", action='store',default=1,required=False)
    parser.add_argument('--unkweight', type=int,help="Unknown character weight for anagram vector representation", action='store',default=1,required=False)
    parser.add_argument('--simple',action='store_true', help="Simple mode: use limited context only (if there is an LM), no n-gram support, do not employ a full beam decoder")
    parser.add_argument('--report',action='store_true', help="Output a full report")
    parser.add_argument('--json',action='store_true', help="Output JSON")
    parser.add_argument('--noout',dest='output',action='store_false', help="Do not output")
    parser.add_argument('-d', '--debug',action='store_true')

def main():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    setup_argparser(parser)
    args = parser.parse_args()
    corrector = Corrector(**vars(args))

    print("Reading test input words from standard input, expecting one word/token per line (if interactively invoked, type ctrl-D when done):",file=sys.stderr)
    testwords = [ w.strip() for w in sys.stdin.readlines() ]
    results = corrector.correct(testwords)
    if args.json:
        corrector.output_json(results)
    elif args.output:
        if args.report:
            if args.simple:
                corrector.output_simple(results)
            else:
                corrector.output_report(results)
        corrector.output(results)

if __name__ == '__main__':
    main()
