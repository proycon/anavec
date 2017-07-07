Anavec
===========

Introduction
-------------

Anavec is a proof-of-concept spelling correction/normalisation system inspired by TICCL  [Reynaert, 2010] and its use of anagram
hashing. Words in a lexicon and background corpus are stored as **anagram vectors**, i.e. an unordered bag-of-characters
model. These are gathered in a matrix and constitute the initial training data.

At testing, words to be corrected are similarly represented as anagram vectors, and gathered in a matrix. Subsequently,
squared pairwise distances are computed between the two matrices, this acts as a simpler heuristic approach to
Levenshtein/edit distance as it can be more efficiently computed. The k-nearest neighbours are retrieved for each test
instance, these represent anagrams and are in turn resolved to all possible correction candidates.

The correction candidates are then ranked according to four components:

* The squared distance between training and test anagram vector
* The actual Levenshtein distance
* The frequency of the candidate in the background corpus
* The presence or absence of the candidate in the lexicon

These components are normalized and summed to form a score for each correction candidate. Each of the four terms is
parametrised by a weight, determining the share it takes in the whole score computation.

This is all context insensitive, but a Language Model can be enabled for context sensitivity. The Language Model will
score sequences of correction candidates in context. A compound score is computed for each possible sequence of
correction candidates, taking into account both the LM score and the score from the correction model.  The weight of the
Language Model versus Correction Model in this computation is again parametrised.

A beam search decoding algorithm subsequently seeks through a large number of correction hypotheses for a given input
sequence, looking for the one that maximises the aforementioned compound score (or returns an n-best list). This decoder
is a stack based decoder as is also common in Machine Translation, meaning that decoding proceeds in a number of stacks,
where each stack aggregates correction hypotheses that cover the same parts of the input sequence. The final stack will
cover the full input sequence.

Anagram vectors can also be computed on n-grams, rather than single words/tokens.

Dependencies
--------------

This software is written in Python 3 and employs various third-party modules implemented in C/C++ for efficient computation:

* `Colibri Core <http://proycon.github.io/colibri-core/>`_ [van Gompel et al, 2016] (for the background corpus)
* `Theano <https://github.com/Theano/Theano>`_ (for multi-dimensional computation, supports GPU)
* `Numpy <http://www.numpy.org>`_
* `Python-Levenshtein <https://github.com/ztane/python-Levenshtein/>`_.
* `KenLM <https://github.com/kpu/kenlm>`_ (for Language Model support)

Installation
---------------

Stable version from the Python Package Index (**not available yet**):

* ``pip3 install anavec``

Latest version from github (https://github.com/proycon/anavec):

* Clone this repository
* ``python3 setup.py install``

The KenLM dependency for LM support needs to be installed separately at it is not in the Python Package Index:

* ``pip3 install https://github.com/kpu/kenlm/archive/master.zip``

Note that ``pip3`` refers to the Python 3 version of ``pip``, it may be available simply as ``pip`` on your system, especially if
you are using a Python Virtual Environment, which we always recommend.

License
----------

GNU Public License v3

Usage
----------

usage: ``anavec.py [OPTIONS] -m PATTERNMODEL -c CLASSFILE``

optional arguments:
  -h, --help            show this help message and exit
  -m PATTERNMODEL, --patternmodel PATTERNMODEL
                        Pattern model of a background corpus (training data;
                        Colibri Core unindexed patternmodel) (default: None)
  -l LEXICON, --lexicon LEXICON
                        Lexicon file (training data; plain text, one word per
                        line) (default: None)
  -L LM, --lm LM        Language model file in ARPA format (default: None)
  -c CLASSFILE, --classfile CLASSFILE
                        Class file of background corpus (default: None)
  -k NEIGHBOURS, --neighbours NEIGHBOURS, --neighbors NEIGHBOURS
                        Maximum number of anagram distances to consider (the
                        actual amount of anagrams is likely higher) (default:
                        3)
  -K CANDIDATES, --candidates CANDIDATES
                        Maximum number of candidates to consider per input
                        token/pattern (default: 100)
  -n TOPN, --topn TOPN  Maximum number of candidates to return (default: 10)
  -N NGRAMS, --ngrams NGRAMS
                        N-grams to consider (max value of n). Ensure that your
                        background corpus is trained for at least the same
                        length for this to have any effect! (default: 3)
  -D MAXLD, --maxld MAXLD
                        Maximum levenshtein distance (default: 5)
  -M MAXVD, --maxvd MAXVD
                        Maximum vector distance (default: 5)
  -t MINFREQ, --minfreq MINFREQ
                        Minimum frequency threshold (occurrence count) in
                        background corpus (default: 1)
  -a ALPHAFREQ, --alphafreq ALPHAFREQ
                        Minimum alphabet frequency threshold (occurrence
                        count); characters occuring less are not considered in
                        the anagram vectors (default: 10)
  -b BEAMSIZE, --beamsize BEAMSIZE
                        Beamsize for the decoder (default: 100)
  --maxdeleteratio MAXDELETERATIO
                        Do not allow a word to lose more than this fraction of
                        its letters (default: 0.34)
  --lexfreq LEXFREQ     Artificial frequency (occurrence count) for items in
                        the lexicon that are not in the background corpus
                        (default: 1)
  --ldweight LDWEIGHT   Levenshtein distance weight for candidating ranking
                        (default: 1)
  --vdweight VDWEIGHT   Vector distance weight for candidating ranking
                        (default: 1)
  --freqweight FREQWEIGHT
                        Frequency weight for candidating ranking (default: 1)
  --lexweight LEXWEIGHT
                        Lexicon distance weight for candidating ranking
                        (default: 1)
  --lmweight LMWEIGHT   Language Model weight for Language Model selection
                        (together with --correctionweight) (default: 1)
  --correctionweight CORRECTIONWEIGHT
                        Correction Model weight for Language Model selection
                        (together with --lmweight) (default: 1)
  --correctscore CORRECTSCORE
                        The score a word must reach to be marked correct prior
                        to decoding (default: 0.6)
  --correctfreq CORRECTFREQ
                        The frequency a word must have for it to be marked
                        correct prior to decoding (default: 200)
  --punctweight PUNCTWEIGHT
                        Punctuation character weight for anagram vector
                        representation (default: 1)
  --unkweight UNKWEIGHT
                        Unknown character weight for anagram vector
                        representation (default: 1)
  --ngramboost NGRAMBOOST
                        Boost unigram candidates that are also predicted as
                        part of larger ngrams, by the specified factor
                        (default: 0.25)
  -1, --simpledecoder   Use only unigrams in decoding (default: False)
  --lmwin               Boost the scores of the LM selection (to 1.0) just
                        prior to output (default: False)
  --locallm             Use a local LM to select a preferred candidate in each
                        candidate list instead of the LM integrated in the
                        decoder (default: False)
  --blocksize BLOCKSIZE
                        Block size: determines the amount of test tokens to
                        process in one go (dimensions of the anavec test
                        matrix), setting this helps reduce memory at the cost
                        of speed (0 = unlimited) (default: 1000)
  --report              Output a full report (default: False)
  --json                Output JSON (default: False)
  --tok                 Input is already tokenized (default: False)
  --noout               Do not output (default: True)
  -d, --debug

