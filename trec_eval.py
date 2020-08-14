from trectools import TrecQrel, TrecRun, TrecEval,procedures


# A typical evaluation workflow
r1 = TrecRun("data/trec_eval/runs/qrels-treceval-clinical_trials-2018-v2.TR")
r1.topics() # Shows the first 5 topics: 601, 602, 603, 604, 605
qrels = TrecQrel("data/trec_eval/qrel/q1_qrels.txt")
r1_p25 = TrecEval(r1, qrels).get_precision(depth=100)

print(r1_p25 )
