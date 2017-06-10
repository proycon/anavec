Anavec
===========

Introduction
-------------

Anavec is a proof-of-concept spelling correction/normalisation system inspired by TICCL  [Reynaert, 2010] and its use of anagram
hashing. Words in a lexicon and background corpus are stored as **anagram vectors**, i.e. an unordered bag-of-characters
model. These are gathered in a matrix and constitutes the initial training data.

At testing, words to be corrected are similarly represented as anagram vectors, and gathered in a matrix. Subsequently,
squared pairwise distances are computed between the two matrices, this acts as simpler heuristic approach to
Levenshtein/edit distance as it can be more efficiently computed. The k-nearest neighbours are retrieved for each test
instance, these represent anagrams and are in turn resolved to all possible correction candidates.

The corrections candidates are then ranked according to four components:
* The squared distance between training and test anagram vector
* The actual Levenshtein distance
* The frequency of the candidate in the background corpus
* The presence of the candidate in the lexicon or not

These components are normalized and summed to form a score for each correction cancidate. Each of the four terms is
parametrised by a weight, determining the share it takes in the whole score computation.

Dependencies
--------------

This software is written in Python 3 and employs various third-party modules implemented in C/C++ for efficient computation:

* `Colibri Core <http://proycon.github.io/colibri-core/>`_ [van Gompel et al, 2016] (for the background corpus)
* `Theano <https://github.com/Theano/Theano>`_ (for multi-dimensional computation, supports GPU)
* `Numpy <http://www.numpy.org>`_
* `Python-Levenshtein <https://github.com/ztane/python-Levenshtein/>`_.

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
  -c CLASSFILE, --classfile CLASSFILE
                        Class file of background corpus (default: None)
  -k NEIGHBOURS, --neighbours NEIGHBOURS, --neighbors NEIGHBOURS
                        Maximum number of anagram neighbours to consider
                        (default: 20)
  -n TOPN, --topn TOPN  Maximum number of candidates to return (default: 10)
  -D MAXLD, --maxld MAXLD
                        Maximum levenshtein distance (default: 5)
  -t MINFREQ, --minfreq MINFREQ
                        Minimum frequency threshold (occurrence count) in
                        background corpus (default: 1)
  -a ALPHAFREQ, --alphafreq ALPHAFREQ
                        Minimum alphabet frequency threshold (occurrence
                        count); characters occuring less are not considered in
                        the anagram vectors (default: 10)
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
  --punctweight PUNCTWEIGHT
                        Punctuation character weight for anagram vector
                        representation (default: 1)
  --unkweight UNKWEIGHT
                        Unknown character weight for anagram vector
                        representation (default: 1)
  --json                Output JSON (default: False)
  -d, --debug
```
