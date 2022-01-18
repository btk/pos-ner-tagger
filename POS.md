You are provided formatted data (in conll-u format) and two complete, ready-to-run, Python
scripts (one uses Logistic Regression and one uses Conditional Random Fields) for the POS
tagging task. Read and try to understand the data and code. Current implementation makes use of
a simple feature set, say Set1 (Set1={token, start_with_capital, has_capitals_inside,
is_all_capitals, has_numbers}). You will design 2 additional feature sets (Set2 and Set3) and
compare performances of 3 feature sets. In order to be able to extract your new features, you will
probably have to add a few lines to the already existing code. However, except the
extract_features() function, its helper, creatdict() function and the main, do not make major
changes in the given code (some other small changes may be required.).

You may add helpers.
To be more specific about what to submit:
1. Skimming through the given data, you will see that it consists of three columns. First
column is tokens of the sentence. What are the other two columns? Which columns do we
use in this task? (Hint: Taking a look to the given code and making a short survey may
help.)

2. Run the tagger_lr_pos.py and report the accuracy, precision, recall, F1 values (of Set1).
Create 2 new feature sets such that they outperform the given Set1. What are your new
feature sets? Report. Also report their accuracies, recalls, precisions and F1 scores.
Submit codes using Set2 and Set3 (as two separate Python scripts named
tagger_lr_pos_set2.py and tagger_lr_pos_set3.py).

3. Repeat step 2 with tagger_crf_pos.py. (You should also report answer to the same
questions, and submit required codes.)

4. Compare two LR and CRF classifiers you have used. Which one is more suitable for this
task, why? Report.
(Hint1: Again, a short survey may help.)
(Hint2: Note that at the training phase classifiers are fitted sentence by sentence instead
of text-by-text etc.)
