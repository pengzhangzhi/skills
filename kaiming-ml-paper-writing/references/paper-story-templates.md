# ML Paper Story Templates

Use a template to test the paper's argument, not to force a label. First test whether the work supports a genuine New paradigm story because the preferred narrative is an intuitive change in how readers see the problem. If it does not, choose the strongest accurate template. Use a secondary one only if both serve the same central claim.

| Template | The paper must prove | The framing breaks when |
|---|---|---|
| Data artifact | The community needs the dataset, benchmark, metric, or protocol; its construction is sound; it reveals a new scientific insight. | The artifact is credible but teaches nothing new. |
| Horse race | The evaluation is legitimate and fair; the win is decisive; a technical reason explains the gain. | The margin is small, the budget differs, or the explanation is absent. |
| New paradigm | The old/new distinction is substantive; an operational method realizes it; the new view unlocks something unavailable before. | The contrast is cosmetic or evaluation still assumes the old paradigm. |
| Resurrected baseline | Prior work accumulated meaningful complexity; a genuinely simpler baseline matches or wins; the paper explains why it was overlooked. | Simplicity is not surprising or comparisons are not controlled. |
| Unification | Apparently different methods fit one framework; important cases are covered; the framework has conceptual or practical payoff. | A central prior method does not fit or no payoff follows. |
| Problem solving | A larger goal is blocked by two or three distinct obstacles; each component addresses one; together they advance the goal. | Problems are vague or a monolithic method is not mapped to them. |
| Discovery | The paper asks a sharp unanswered question; the answer is credible; the implication changes scientific understanding or practice. | The question is vague or the evidence is merely exploratory. |
| Countervailing wisdom | A real community belief is stated fairly; evidence overturns it; the paper gives a corrected view and explains the earlier mistake. | The belief is a straw man or the correction is no more predictive. |

## Convert the template into evidence

Assign every experiment one primary job:

- **Existence:** Does the claimed effect occur?
- **Mechanism:** Is the proposed explanation supported?
- **Control:** Isolate the causal design choice or rule out an alternative.
- **Boundary:** Where does the effect weaken or fail?
- **Comparison:** Does it matter against fair baselines?
- **Generalization:** Does it transfer across data, tasks, models, or scale?
- **Cost:** What compute, data, latency, or complexity buys the result?

If an experiment has no job, remove it or move it to an appendix. If a central claim has no experiment, narrow the claim or plan the missing test.

Before adding an experiment to the main paper, complete this sentence: “This experiment shows ___ because ___ is held fixed while ___ changes.” If the sentence is unclear, redesign the experiment or move it out of the main story.

## Common coherent combinations

- New paradigm + Horse race: the empirical win validates the new view; the view remains primary.
- Resurrected baseline + Discovery: simplicity exposes which component actually matters.
- Unification + Problem solving: the framework yields a concrete algorithm or implementation benefit.
- Countervailing wisdom + Discovery: a sharp question corrects an accepted explanation.

Do not combine templates merely to enlarge the contribution list.
