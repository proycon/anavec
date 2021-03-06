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
from anavec.anavec import Corrector, setup_argparser, InputTokenState, readinput

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


def setbreakpoints(testtokens, mask, blocksize, eager=False):
    """input is all on one line, this will overwhelm the decoder, split into 'lines' at points where punctuation likely indicates a sentence ending"""
    begin = 0
    blockbegin = 0
    for i, (testtoken, state) in enumerate(zip(testtokens, mask)):
        if testtoken == '.' or (eager and testtoken[-1] == '.' and i+1 < len(testtokens) and mask[i+1] & InputTokenState.CORRECT):
            if i - begin >= 6 and i+1 < len(testtokens) and testtokens[i+1][0].isalpha() and testtokens[i+1][0] == testtokens[i+1][0].upper():
                mask[i] |= InputTokenState.EOL
                begin = i
                if i - blockbegin >= blocksize:
                    mask[i] |= InputTokenState.EOB
                    blockbegin = i

def process(corrector, testfiles, args):
    icdar_results = {} #results as per challenge specification

    if args.positionfile:
        positiondata = readpositiondata(args.positionfile)
    elif args.task == 2:
        raise Exception("No position file specified, required for task 2!")

    for testfile in testfiles:
        print(" === PROCESSING " + testfile + " === ",file=sys.stderr)
        icdar_results[testfile] = {} #results as per challenge specification

        text = loadtext(testfile)

        lines = text.strip('\n').split('\n') #should actually only split into one item for this task
        if len(lines) != 1:
            raise Exception("Input file " + testfile + " contains more lines! Invalid according to specification!")

        testtokens, mask, positions = readinput(lines, False, args.blocksize)

        #input is all on one line, this will overwhelm the decoder, split into 'lines' at points where punctuation likely indicates a sentence ending
        setbreakpoints(testtokens, mask, args.blocksize, eager=True)

        if args.positionfile:
            if testfile not in positiondata:
                print("WARNING: Testfile " + testfile + " is not found in the position data! Skipping!", file=sys.stderr)
                continue
            elif not positiondata[testfile]:
                #found but empty (happens it seems, just move on to the next testfile
                print("WARNING: Testfile " + testfile + " exists but has no positions the position data! Skipping!", file=sys.stderr)
                continue
            refpositions = { int(positiontuple.split(':')[0]): int(positiontuple.split(':')[1]) for positiontuple in positiondata[testfile] }
        else:
            refpositions = {}
        foundpositions = {} #bookkeeping for task 2

        assert len(testtokens) == len(mask)
        assert len(mask) == len(positions)
        if args.positionfile:
            #prepare merged data structures, we will copy everything into this and merge tokens that are to be treated as one according to the reference positions
            testtokens_merged = []
            mask_merged = []
        mergelength = 0
        positions_merged = [] #this one we use always (converts 3-tuple to 4-tuples by adding tokenlength explicitly, even if it's just 1 for all )

        for i, (token, state, (beginchar, endchar,punctail)) in enumerate(zip(testtokens, mask, positions)):
            if beginchar is None:
                #token is trailing punctuation
                print("       input token #" + str(i) + "           (trailing punctuation) -> " + punctail,file=sys.stderr)
                if args.positionfile and mergelength == 0:
                    testtokens_merged.append(token)
                    mask_merged.append(state)
                if mergelength == 0: positions_merged.append( (beginchar, 1, endchar, punctail) )
            else:
                if beginchar in refpositions:
                    print("  REFERENCED token #" + str(i) + " (l=" + str(refpositions[beginchar]) + ") " + testfile + "@" + str(beginchar) + ":1 " + text[beginchar:endchar] + " -> " + token,end="",file=sys.stderr)
                    if args.positionfile:
                        mask[i] |= InputTokenState.INCORRECT #force correction
                        state = mask[i]
                        foundpositions[beginchar] = True
                        if refpositions[beginchar] > 1:
                            token += punctail #consume trailing punctuation as part of token, it's no longer trailing
                            mergelength = refpositions[beginchar]-1

                            print("\t[MERGING WITH NEXT " + str(mergelength)+" AS ",end="",file=sys.stderr)
                            origoffset = 1
                            offset = 1
                            while origoffset <= mergelength:
                                beginchar2, endchar2, punctail2 = positions[i+offset]
                                if beginchar2 is None:
                                    #trailing punctuation
                                    token += testtokens[i+offset]
                                else:
                                    token += " " + testtokens[i+offset]
                                    if origoffset == mergelength:
                                        punctail = punctail2
                                        endchar = endchar2
                                    else:
                                        token += punctail2
                                    origoffset += 1
                                offset += 1
                            print(token + "]",end="",file=sys.stderr)

                        testtokens_merged.append(token)
                        state = InputTokenState.CORRECTABLE
                        mask_merged.append(state)
                        positions_merged.append( (beginchar, refpositions[beginchar], endchar, punctail) )
                        print("\t[MERGE SPANS CHARS " + str(beginchar)+" TO " + str(endchar) ,end="",file=sys.stderr)
                else:
                    print("       input token #" + str(i) + "       " + testfile + "@" + str(beginchar) + ":1 " + text[beginchar:endchar] + " -> " + token,end="",file=sys.stderr)
                    if args.positionfile:
                        if mergelength > 0:
                            mergelength -= 1
                            print("\t[MERGED]",file=sys.stderr)
                            continue

                        #mark this token as correct (it's not in the positions file)
                        mask[i] |= InputTokenState.CORRECT #force correction
                        state = mask[i]

                        testtokens_merged.append(token)
                        mask_merged.append(state)
                    positions_merged.append( (beginchar, 1, endchar, punctail) )

                if punctail:
                    print("\t[punctail=" + punctail + "]",end="",file=sys.stderr)
                if state & InputTokenState.CORRECT:
                    print("\t[KEEP]",end="",file=sys.stderr)
                elif state & InputTokenState.INCORRECT:
                    print("\t[FORCE-PROCESS]",end="",file=sys.stderr)
                print(file=sys.stderr)
            if state & InputTokenState.EOL:
                print("  Tokenisation --eol--",file=sys.stderr)
            if state & InputTokenState.EOB:
                print("  Tokenisation --end of block--",file=sys.stderr)

        for beginchar in sorted(refpositions):
            if beginchar not in foundpositions:
                print("WARNING: Position @" + str(beginchar) + ":" + str(refpositions[beginchar]) + " was not found in " + testfile,file=sys.stderr)

        if args.task not in (1,2): continue

        if args.positionfile: #task 2
            testtokens = testtokens_merged
            mask = mask_merged
        positions = positions_merged
        foundpositions = {}
        assert len(testtokens) == len(mask)
        assert len(mask) == len(positions)

        for results in corrector.correct(testtokens, mask):
            print("Corrector input: ", " ".join(results.testtokens),file=sys.stderr)
            print("Corrector best output: ", str(results.top[0]),file=sys.stderr)

            if args.options == 0:
                print("(Primary source is decoder output)",file=sys.stderr)
                for candidate in results.top[0]:
                    index = results.offset + candidate.hypothesis.index
                    beginchar, origtokenlength, endchar, punctail = positions[index]
                    if beginchar is None:
                        #ignore trailing punctuation
                        continue

                    if candidate.error or (args.positionfile and beginchar in refpositions):
                        tokenlength = candidate.hypothesis.length #in tokens
                        correction = candidate.text

                        if args.positionfile:
                            if beginchar not in refpositions or refpositions[beginchar] != origtokenlength:
                                #task 2: not a reference token, ignore
                                continue
                            else:
                                foundpositions[beginchar] = True

                        #re-add any trailing punctuation
                        correction += punctail
                        original = text[beginchar:endchar]
                        print(" Correction [" + testfile + "@" + str(beginchar) + ":" + str(origtokenlength) + "] " + original + " -> " + correction, file=sys.stderr)
                        icdar_results[testfile][str(beginchar)+":"+str(origtokenlength)] = { correction: candidate.score }
            else:
                print("(Primary source is candidate tree)",file=sys.stderr)
                for index in sorted(results.candidatetree):
                    globalindex = results.offset + index
                    beginchar, origtokenlength, endchar, punctail = positions[globalindex]
                    if beginchar is None:
                        #ignore trailing punctuation
                        continue

                    if args.positionfile:
                        if beginchar not in refpositions:
                            continue
                        foundpositions[beginchar] = True

                    if 1 in results.candidatetree[index]:
                        candidates = list(sorted(results.candidatetree[index][1], key=lambda x: (x.lmselect * -1, x.logprob * -1)))[:args.options]
                        if candidates:
                            #scoresum = sum( (candidate.score for candidate in candidates ) )
                            original = text[beginchar:endchar]
                            print(" Correction [" + testfile + "@" + str(beginchar) + ":" + str(origtokenlength) + "] " + original + " -> " + "; ".join([ candidate.text + " (" + str(10**candidate.logprob) + ") " for candidate in candidates]) + " [punctail=" + punctail+"]", file=sys.stderr)
                            icdar_results[testfile][str(beginchar)+":"+str(origtokenlength)] = { candidate.text + punctail: 10**candidate.logprob for candidate in candidates }



        for beginchar in sorted(refpositions):
            if beginchar not in foundpositions:
                print("WARNING: Position @" + str(beginchar) + ":" + str(refpositions[beginchar]) + " was not corrected in " + testfile,file=sys.stderr)


    return icdar_results



def main():
    parser = argparse.ArgumentParser(description="ICDAR 2017 Post-OCR Correction Processing Script for Task 2 with Anavec", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help="Input file or directory (*.txt files)", action='store',required=True)
    parser.add_argument('--positionfile', type=str, help="Input file with position information (erroneous_tokens_pos.json), required for task 2", action='store',required=False)
    parser.add_argument('--options', type=int, help="Maximum number of options to output, if set to 0 (default), the best option according to the decoder will outputted. For values higher than 0, the candidate tree will be explicitly consulted instead, which limits the use of ngrams", action='store',default=0)
    parser.add_argument('--task', type=int, help="Task", action='store',required=True)
    setup_argparser(parser) #for anavec
    args = parser.parse_args()
    args.lmwin = True

    if os.path.isdir(args.input):
        testfiles = []
        for f in glob.glob(args.input + "/*.txt"):
            testfiles.append(f)
    else:
        testfiles = [args.input]
    print("Found testfiles:", testfiles,file=sys.stderr)

    if args.task > 0:
        corrector = Corrector(**vars(args))
    else:
        corrector = None
    results = process(corrector, testfiles, args)
    #if args.task == 1:
    #    results = process_task1(corrector, testfiles, args)
    #elif args.task == 2:
    #    results = process_task2(corrector, testfiles, args.positionfile, args)
    #else:
    #    raise NotImplementedError

    #Output results as JSON to stdout
    print(json.dumps(results, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()
