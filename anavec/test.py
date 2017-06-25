#!/usr/bin/env python3

#-------------------------------------------------------------------
# Test/evaluation script
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
from pynlpl.evaluation import ClassEvaluation

def main():
    parser = argparse.ArgumentParser(description="Anavec test and evaluation script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--referencefile', type=str, help="Reference list (particular format)", action='store',required=True)
    setup_argparser(parser) #for anavec
    args = parser.parse_args()


    references = []
    alltestwords = []
    allmask = []
    correct = 0
    with open(args.referencefile, 'r',encoding='utf-8') as f:
        windowmask = [InputTokenState.CORRECT]*2 + [InputTokenState.CORRECTABLE] + [InputTokenState.CORRECT, InputTokenState.CORRECT | InputTokenState.EOL ]
        for line in f:
            fields = line.strip().split('#')
            error = fields[0]
            correction = fields[1]
            context = fields[2].split(' ')
            windowwords = context[:2] + [error] + context[2:]
            alltestwords += windowwords
            allmask += windowmask
            references.append( (windowwords, windowmask, correction) )

    l =  len(references)
    print("Read " + str(l) + " reference instances",file=sys.stderr)

    assert len(alltestwords) == len(allmask)

    corrector = Corrector(**vars(args))

    observations = []
    goals = []
    for result, (testwords, mask, reference) in zip(corrector.correct(alltestwords, allmask), references):
        output = list(sorted(result['candidatetree'][2][1], key=lambda x: -1 * x.score))
        if output:
            if reference == output[0].text:
                match = "BEST"
            elif reference in [c.text for c in output]:
                match = "YES"
            else:
                match = "NO"
            print(match + "\t[" + reference + "]\t" +  testwords[2] + "\t-->\t" + "\t".join([ c.text + "["+str(c.score)+"]" for c in output]))
            output = output[0].text
        else:
            output = ""
            print("NO\t[" + reference + "]\t" + testwords[2] + "\t-->\tNO-SUGGESTIONS! [0.0]")
        observations.append(output)
        goals.append(reference)

    evaluation = ClassEvaluation(goals, observations)
    print("# Precision = ", evaluation.precision())
    print("# Recall = ", evaluation.recall())

if __name__ == '__main__':
    main()




