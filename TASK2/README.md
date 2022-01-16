Similar to TASK 1, however, the task is NER instead of POS-tagging. Again you will run the
existing code which includes feature set 1 and report metrics. Then, again you will create 2 new
feature sets specific to this task, report those sets, their metrics and submit your codes. Except
the extract_features() function, its helper, creatdict() function and the main, do not make major
changes in the given code (some other small changes may be required.). You may add helpers.
The data you will work on employs IOB encoding. If you do not remember that encoding,
starting with remembering it by studying the course material may be helpful.

Training data includes additional tags (i.e. POS-tags) but test data do not. Therefore, at the
testing phase, POS-tagger of nltk is used. If you need an additional tagger for this task, use nltk’s
taggers. If that is the case, you are allowed (and have to) properly add this tagger to the test
function. If you do that, indicate in your report

To be more specific about what to submit:

1. Run the tagger_lr_ner.py and and report the accuracy, precision, recall, F1 values (of
Set1). Create 2 new feature sets such that they outperform the given Set1. What are your
new feature sets? Report. Also report their accuracies, recalls, precisions and F1 scores.
Submit codes using Set2 and Set3 (as two separate Python scripts named
tagger_lr_pos_set2.py and tagger_lr_pos_set3.py).

2. Repeat step 2 with tagger_crf_ner.py. (You should also report answer to the same
questions, and submit required codes.)

3. What can be done to improve the performance of LR such that it becomes closer to that
of CRF? (Hint: Think about the difference between CRF and LR. May adding some type
of features reduce the difference?)

4. You will observe that both precision and recall values are remarkably smaller than the
corresponding accuracy value. This is due to the testing methodology. What is this testing
methodology; why is it needed? (Hint: Investigating the test function and making a short
survey will be helpful.)
