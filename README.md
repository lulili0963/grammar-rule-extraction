# grammar-rule-extraction

This is the master internship work in LISN.

The objective of this internship is to extract agreement rules, one of the grammatical phenomena, via interpretable generative modeling.

For the whole work, there are 5 steps.

First, we construct our own training data through SUD treebanks. We only focus on French and English corpora.

Second, we adapt the basic Sigmoid Belief Network to the training data because we have discrete data in the training dataset.

Third, we use proximal methods to train our SBN model to get a sparse model. In order to achieve this goal, we use L1-norm during the training process.

Then, we propose a practical method to interpret the parameters learned by the model. 

Finally, we extract agreement rules and use a statistical test to evaluate them.

