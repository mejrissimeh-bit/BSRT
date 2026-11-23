# Technical description

BSRT maps a context C and removal fraction r to an ordered subsequence of its
WordPiece tokens. The frozen BERT encoder receives C without a question. A
trainable two-layer Transformer combines contextual states, positions,
structural token features, and a learned continuous budget embedding to produce
one retention logit s_i(C,r) per token.

Whole WordPiece groups are the indivisible selection units. A knapsack solver
maximises the sum of selected token logits subject to retaining
N - floor(rN) eligible WordPieces, or the closest feasible count if grouping
prevents that exact count. Encoder-inserted CLS/SEP tokens are outside the
selection sequence. A source span tokenized as UNK is treated as a source token,
as in the previous learned policy. Retained tokens keep their original order. Character offsets reconstruct the selected
text; the model-token count after reconstruction is measured separately.

Answer-bearing pretraining uses the union of annotated answer spans from each
training context. The target t_i is one when token i overlaps any training
answer span, and zero otherwise. Positive-class weighting limits the impact of
the large number of non-answer tokens. A soft penalty encourages the mean
retention probability to match 1-r. These labels train the model but are absent
from its inference interface.

Offline search evaluates hard masks M by mean QA token-F1 across the selected
training questions of C. Candidate masks are generated from retention logits
with stochastic Gumbel exploration and a whole-word budget solver. A
cross-entropy-method proposal update uses the retention frequency among elite
masks; equal-sized group swaps provide local refinement. The frozen QA model
evaluates the same compressed context for every question. QA predictions are
cached by full text, question, model revision, and inference settings.

For distillation, each elite mask receives a weight proportional to
exp((reward - maximum reward)/temperature). Their weighted retention average
forms a soft token target. Binary cross-entropy against that target is combined
with the answer-bearing auxiliary objective and budget penalty. This is a
surrogate training loss, so validation selection evaluates actual hard deletion
through the downstream QA system.

After each improvement round the best validation checkpoint proposes another
training-only search round. Earlier elite targets remain in the distillation
dataset. The selected deployable checkpoint is the one with highest mean
context QA F1 on separate validation articles, across the configured budgets.
If distillation is less effective than answer-span pretraining, the earlier
checkpoint remains selected. Performance improvements and their absence are
both reported from actual evaluation artifacts.

The final 500 historical SQuAD v1.1 validation pairs are excluded from all new
training and checkpoint selection. An additional article-disjoint test set is
reserved from SQuAD2 training for a separate assessment of compressor
generalisation. The QA teacher is already fine-tuned, so the training-corpus
split establishes independence for the compressor rather than for the teacher.

At deployment the model exposes only compress(context, removal_ratio).
Questions and answers are used exclusively in offline training supervision or
downstream evaluation. The deployed compressor performs no QA inference and
no reward-driven mask search.
