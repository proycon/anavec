#!/usr/bin/env python3

#-------------------------------------------------------------------
# Processing script for ICDAR 2017 Post-OCR Text Correction challenge
# Uses anavec
#-------------------------------------------------------------------
#       by Maarten van Gompel
#       Radboud University Nijmegen
#       Licensed under GPLv3

import os
import sys
import argparse
import glob
import json
from anavec import anavec

def loadtext(testfile):
    """Load the text from a test file"""
    with open(testfile,'r',encoding='utf-8') as f:
        for line in f:
            if line.startwith("[OCR_toInput]"):
                return line[len("[OCR_toInput]") + 1:]
            else:
                raise Exception("Unexpected input format, expected [OCR_toInput] on first line")

    raise Exception("No text found")

def readpositiondata(positionfile):
    with open(positionfile,'r',encoding='utf-8') as f:
        positiondata = json.load(f)
    return positiondata

def process_task2(testfiles, positionfile, args):
    positiondata = readpositiondata(positionfile)

    icdar_results = {} #results as per challenge specification

    for testfile in testfiles:
        if testfile not in positiondata:
            raise Exception("Test file " + testfile + " does not occur in the position data file!")
        positions = [ (int(positiontuple.split(':')[0]), int(positiontuple.split(':')[1])) for positiontuple in positiondata[testfile] ]
        icdar_results[testfile] = {} #results as per challenge specification

        text = loadtext(testfile)
        tokens = text.strip(' ')

        result = {} #this will store the results

        #greedy match over all 3,2,1-grams, in that order
        charoffset = 0
        testwords = []
        for i, token in enumerate(tokens):
            for position_charoffset, position_tokenlength in positions:
                if charoffset == position_charoffset:
                    testword = tokens[i:i+position_tokenlength]
                    testwords.append(testword)

            charoffset += len(token)

        if len(testwords) != len(positions):
            raise Exception("One or more positions were not found in the text!")


        results = anavec.run(*testwords, **vars(args)) #results as presented by anavec

        for result, testword, (charoffset, tokenlength) in zip(results, testwords, positions):
            assert result.text == testword
            icdar_results[testfile][str(charoffset)+":"+str(tokenlength)] = { candidate.text: candidate.score for candidate in result.candidates }

    return icdar_results


def main():
    parser = argparse.ArgumentParser(description="ICDAR 2017 Post-OCR Correction Processing Script for Task 2 with Anavec", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--input', dest='settype',type="str", help="Input file or directory (*.txt files)", action='store',required=True)
    parser.add_argument('--positionfile', dest='settype',type="str", help="Input file with position information (erroneous_tokens_pos.json)", action='store',required=True)
    anavec.setup_argparser(parser)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        testfiles = []
        for f in glob.glob(args.input + "/*.txt"):
            testfiles.append(f)
    else:
        testfiles = [args.input]

    results = process_task2(testfiles, args.positionfile, args)

    #Output results as JSON to stdout
    print(json.dumps(results))

if __name__ == '__main__':
    main()
