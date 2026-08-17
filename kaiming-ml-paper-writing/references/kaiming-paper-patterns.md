# Paper-Level Patterns from a Kaiming He Corpus

## Scope

These are high-level patterns observed in papers Kaiming He coauthored. Coauthorship does not establish who wrote a sentence, so use the findings as corpus-level guidance rather than personal imitation. Do not copy wording.

The corpus spans early vision work through recent representation and generative modeling: [Dark Channel Prior](https://people.csail.mit.edu/kaiming/publications/cvpr09.pdf), [SPP-net](https://arxiv.org/abs/1406.4729), [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852), [ResNet](https://arxiv.org/abs/1512.03385), [Identity Mappings](https://arxiv.org/abs/1603.05027), [Mask R-CNN](https://arxiv.org/abs/1703.06870), [Group Normalization](https://arxiv.org/abs/1803.08494), [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883), [MoCo](https://arxiv.org/abs/1911.05722), [SimSiam](https://arxiv.org/abs/2011.10566), [MAE](https://arxiv.org/abs/2111.06377), [autoregressive generation without vector quantization](https://arxiv.org/abs/2406.11838), [MeanFlow](https://arxiv.org/abs/2505.13447), and [Back to Basics](https://arxiv.org/abs/2511.13720). The [official publication page](https://people.csail.mit.edu/kaiming/) anchors the selection.

## Stable patterns

### 1. One surprise, compressed into one primitive

The story usually turns on a single reusable concept: a prior, residual function, aligned mask branch, dynamic dictionary, stop-gradient, asymmetric masking, continuous-token loss, average velocity, or clean-data prediction. The abstract names it early; the introduction motivates it; the method formalizes it; the experiments test it.

**Use:** Force a one-sentence central claim and one named primitive. If two concepts compete, decide which explains the other.

### 2. Context, tension, question, answer

Introductions begin with an accepted success or practice, identify a precise limitation or questionable assumption, and often pose a crisp question. The proposed method follows as a response to that diagnosis, not as an isolated list of components.

**Use:** Make the old/new contrast exact. Explain why the limitation exists before describing the fix.

### 3. Concept before machinery

An intuitive reinterpretation often precedes equations: contrastive learning as dictionary lookup, residual learning instead of direct mapping, or one-step generation through average rather than instantaneous velocity. Notation is introduced only after the reader knows what it is for.

**Use:** Give the reader the mental model, then the formal object, then the implementation.

### 4. Experiments follow the skeptical reader's questions

The page-one figure often shows the surprising headline. The experiment body then establishes credibility through a controlled property study, a fair main comparison, causal ablations, broader transfer or scale, and a failure boundary. The exact order varies: some papers lead with the main result, while others establish the mechanism first. Subsections correspond to questions or design axes, and observations explicitly return to the hypothesis.

**Use:** Order evidence as: why care, why believe, how broad, and where it fails. Put each result where it answers the reader's next objection.

### 5. Fairness and negative evidence are part of the story

Comparisons state what is held constant, distinguish reproduced from reported results, and disclose exceptions. MoCo reports a task where it lags; SimSiam distinguishes collapse from other failures; MAE says when differences are statistically insignificant.

**Use:** Report the strongest counterexample that changes the scope. A credible boundary strengthens the central claim.

Choose controls that are favorable to the incumbent whenever possible. Matched architecture, compute, schedule, data, and evaluation let the prose make causal statements without rhetorical inflation.

### 6. Claims separate evidence from interpretation

Direct findings use concrete numbers and scoped statements. Mechanistic explanations are often marked as hypotheses or suggestions. Broad implications are saved until evidence has accumulated.

**Use:** Write the observation first, then make the strongest coherent interpretation. Do not hedge a directly supported claim. When a bold framing exceeds the current test, keep the prose clean and flag the gap separately for the author.

### 7. Conclusions are short and forward-looking

Conclusions rarely replay a long contribution list. They extract the conceptual lesson, name an open question or limitation, and indicate what the idea might enable next.

**Use:** End on what the community should reconsider, not on another leaderboard summary.

### 8. Prose is direct, structural, and quantitative

Paragraphs announce their job. Enumerations map real distinctions. Technical terms remain stable. Transitions such as contrast, consequence, and qualification carry the logic. Adjectives are usually backed by a number, controlled comparison, or design fact.

**Use:** Prefer “A reduces B by 3x under C” over generic claims of effectiveness. Keep caveats next to the claim they qualify.

### 9. Readability comes from argument order

The papers feel easy to read because prerequisites arrive just before they are needed, one paragraph usually performs one logical move, and figures or experiments carry part of the explanation. Simplicity is structural, not merely a preference for short sentences.

**Use:** Order material as intuition, precise claim, minimal formalism, controlled evidence, then implication. Delete a detail from the main text if it does not change the reader's mental model.

### 10. Simplicity exposes one critical detail

The high-level intervention is often minimal, but the papers identify the technical detail that makes it work: alignment in Mask R-CNN, the normalization axis in Group Normalization, sufficient convergence when training from scratch, or stop-gradient in SimSiam.

**Use:** State both truths: the overall idea is simple, and one precise detail is essential. Test that detail directly.

### 11. Two figures can carry the whole story

Opening figures commonly expose the surprising result or failure, while the method diagram isolates the changed component. The strongest captions contain the experimental contract and takeaway instead of merely naming panels.

**Use:** Start with a teaser that delivers the result hook, then a method figure that makes the primitive intuitive. Write their one-sentence messages before the surrounding prose.

## Research philosophy that sharpens the story

In his 2024 talk [ML Research, via the Lens of ML](https://people.csail.mit.edu/kaiming/neurips2024workshop/neurips2024_newinml_kaiming.pdf), He emphasizes surprise, simplicity, pre-hoc prediction, future generalization, and scalability. Translate those themes into five checks:

1. What result would surprise an informed reader?
2. What did we predict before seeing the final experiments?
3. What is the simplest explanation consistent with the evidence?
4. What should remain true outside the current benchmark or configuration?
5. Does the idea become more useful, not less, as data, models, or compute scale?

## Final drafting check

- Can a reader state the problem, surprise, and primitive after the abstract?
- Does each method component answer a named obstacle?
- Does each central claim have a controlled test?
- Are result and interpretation visibly separate?
- Is the strongest limitation stated near the affected claim?
- Does the conclusion say what was learned rather than repeat what was built?
