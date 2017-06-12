#!/usr/bin/env python3

#-------------------------------------------------------------------
# Processing script for ICDAR 2017 Post-OCR Text Correction challenge
# Uses anavec
#-------------------------------------------------------------------
#       by Maarten van Gompel
#       Radboud University Nijmegen
#       Licensed under GPLv3

import os
import argparse
import sys
import glob
import json
from anavec.anavec import Corrector, setup_argparser, InputTokenState

def loadtext(testfile):
    """Load the text from a test file"""
    with open(testfile,'r',encoding='utf-8') as f:
        for line in f:
            if line.startswith("[OCR_toInput]"):
                return line[len("[OCR_toInput]") + 1:]
            else:
                raise Exception("Unexpected input format, expected [OCR_toInput] on first line")

    raise Exception("No text found")

def readpositiondata(positionfile):
    with open(positionfile,'r',encoding='utf-8') as f:
        positiondata = json.load(f)
    return positiondata

def process_task2(corrector, testfiles, positionfile, args):
    positiondata = readpositiondata(positionfile)

    icdar_results = {} #results as per challenge specification

    for testfile in testfiles:
        print(" === PROCESSING " + testfile + " === ",file=sys.stderr)
        if testfile not in positiondata:
            raise Exception("Test file " + testfile + " does not occur in the position data file!")
        positions = [ (int(positiontuple.split(':')[0]), int(positiontuple.split(':')[1])) for positiontuple in positiondata[testfile] ]
        icdar_results[testfile] = {} #results as per challenge specification

        text = loadtext(testfile)
        tokens = text.split(' ')

        result = {} #this will store the results

        #greedy match over all 3,2,1-grams, in that order
        charoffset = 0
        testwords = []
        mask = []
        found = 0
        skip = 0
        positions_extended = []
        for i, token in enumerate(tokens):
            if skip: #token was already processed as part of multi-token expression, skip:
                skip -= 1
                continue
            state = InputTokenState.CORRECT #default state is CORRECT, only explicitly marked tokens will be correctable
            #Do we have an explicitly marked token here?
            for position_charoffset, position_tokenlength in positions:
                if charoffset == position_charoffset:
                    token = " ".join(tokens[i:i+position_tokenlength])
                    skip = position_tokenlength -1 #if the token consists of multiple tokens, signal to skip the rest
                    print("[" + testfile + "@" + str(position_charoffset) + ":" + str(position_tokenlength) + "] " +  token, file=sys.stderr)
                    state = InputTokenState.INCORRECT
                    positions_extended.append((position_charoffset,position_tokenlength))
                    found += 1
            if state != InputTokenState.INCORRECT: positions_extended.append((0,0)) #placeholder for words that are not in the position file (already
            mask.append(state)
            testwords.append(token)
            charoffset += len(token) + 1

        if found != len(positions):
            raise Exception("One or more positions were not found in the text!")


        print("Running anavec on input: ", list(zip(testwords,mask, positions_extended)), file=sys.stderr)
        results = corrector.correct(testwords, mask) #results as presented by anavec

        for result, testword, (charoffset, tokenlength) in zip(results, testwords, positions_extended):
            assert result.text == testword
            if tokenlength > 0: #tokenlength == 0 occurs as a placeholder to signal tokens that were not in the position file
                icdar_results[testfile][str(charoffset)+":"+str(tokenlength)] = { candidate.text: candidate.score for candidate in result.candidates }

    return icdar_results


def main():
    parser = argparse.ArgumentParser(description="ICDAR 2017 Post-OCR Correction Processing Script for Task 2 with Anavec", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help="Input file or directory (*.txt files)", action='store',required=True)
    parser.add_argument('--positionfile', type=str, help="Input file with position information (erroneous_tokens_pos.json)", action='store',required=True)
    setup_argparser(parser) #for anavec
    args = parser.parse_args()

    if os.path.isdir(args.input):
        testfiles = []
        for f in glob.glob(args.input + "/*.txt"):
            testfiles.append(f)
    else:
        testfiles = [args.input]

    corrector = Corrector(**vars(args))
    results = process_task2(corrector, testfiles, args.positionfile, args)

    #Output results as JSON to stdout
    print(json.dumps(results, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()
