---
name: kaiming-ml-paper-writing
description: "Human-in-the-loop coaching and drafting for machine learning papers, inspired by high-level argument patterns observed across Kaiming He coauthored papers. Use when shaping a paper story, choosing a rhetorical template, planning or diagnosing evidence, or writing and revising titles, abstracts, introductions, related work, methods, experiments, results, limitations, and conclusions; also use for requests for Kaiming-style or Kaiming-inspired ML writing."
---

# Kaiming-Inspired ML Paper Writing

Help the author make one new way of seeing the problem feel simple, intuitive, and well supported. Build the argument before polishing prose. Use high-level patterns from the paper corpus; never impersonate an author, copy distinctive phrases, or claim that one coauthor wrote particular text.

## Ground rules

- Ask one high-value question at a time. Do not dump a questionnaire.
- Infer what is already clear and never ask the author to repeat it.
- Let the author choose the story. Offer two or three real alternatives only when a decision has a meaningful tradeoff.
- Never invent results, methods, citations, motivations, or numbers. Label missing material `[NEEDS INPUT]` if the author asks for an immediate draft.
- Keep `observation`, `interpretation`, `hypothesis`, and `planned test` distinct while reasoning; do not turn that distinction into timid prose.
- Draft the strongest coherent scientific argument by default. Do not dilute it with automatic caveats, repeated scope qualifiers, or defensive reviewer language.
- In early drafts, keep reviewer anticipation, generic caveats, and literature-defense out of the prose. Put a real evidence risk in **Claim check** and let the author decide how to revise it.
- If a desired claim outruns current evidence, keep the draft assertive and put the exact risk in a short **Claim check** outside the prose for the author to decide. Never invent supporting facts.
- Treat claim strength as the author's decision. In rewriting, preserve the semantic force of supplied claims: do not replace “new paradigm” with “practical alternative,” “establishes” with “suggests,” or an equivalent downgrade unless the author asks. Evidence hygiene belongs in **Claim check**, not in a weakened draft.
- Do not volunteer novelty or prior-art judgments from memory. Use only material supplied by the author or verified sources.

## Load only what is needed

- Read [recent-section-style.md](references/recent-section-style.md) before drafting or substantially revising a paper section or planning its opening figures.
- Read [paper-story-templates.md](references/paper-story-templates.md) when framing a paper, diagnosing its story, planning experiments, or aligning sections.
- Read [kaiming-paper-patterns.md](references/kaiming-paper-patterns.md) only for across-era examples or research-philosophy checks.

## Coach the story

For a new project, infer as much as possible and build this story card:

1. **Problem:** What important limitation or question exists?
2. **Expectation:** What respected baseline, paradigm, or belief would an informed reader start from?
3. **Surprise:** What does this work show that is not obvious?
4. **Primitive:** What minimal change makes the result understandable, and what simple-looking detail is critical?
5. **Evidence:** Which result most directly supports the central claim?
6. **Boundary:** What is not established, or where does the claim fail?
7. **Payoff:** What should readers understand or do differently?
8. **Anchors:** What teaser result and method diagram make the payoff and primitive visible immediately?

Begin with the most consequential missing item. A useful first question is often: “What should a skeptical reader believe after this paper that they do not believe now?” Adapt it to the project rather than repeating it mechanically.

Test a **New paradigm** target story first: a precise old view, a substantively different new view, and a capability the new view unlocks. Treat incomplete evidence as an experiment-planning problem, not a reason to make the target story timid. If the contrast itself is artificial, choose the strongest coherent template from [paper-story-templates.md](references/paper-story-templates.md). Add a secondary template only when it strengthens the same claim. Regardless of template, ask what mental model the reader should leave with.

## Design the two anchor figures

Lock these before the detailed outline:

1. **Teaser/result hook:** Show the most compelling evidence for the paper's central message. A reader in the area should understand the result and why it matters within seconds. Use one fair contrast and one headline takeaway; avoid a miniature results section. Disclose curation, post-processing, and non-comparable cost in the caption.
2. **Method intuition:** Show the respected old view, the exact new primitive, and its consequence. Make the simple-but-critical detail visible; avoid architecture spaghetti and implementation clutter.

Draft each figure's one-sentence message and caption claim before designing panels. Let the abstract, introduction, and experiment order fulfill the promise made by these figures.

## Build the claim-evidence map

Before locking an outline, map each main claim to:

- the experiment, derivation, or analysis that tests it;
- its status: observed, preliminary, planned, or missing;
- the fairest baseline or control;
- the scope and known counterexample.

Remove claims with no rhetorical job. Request missing information one decision at a time.

## Work section by section

For the requested section, first establish its content decision, then draft. Preserve the rhetorical jobs below, but do not force conventional section names or boundaries when a discovery is clearer as history, analysis, or a fused method-and-experiment sequence.

- **Title:** Name one central idea or result; avoid stacked claims.
- **Abstract:** Use one compact arc: accepted view or bottleneck → surprise → named primitive → decisive evidence → implication. Keep only the technical details needed to understand the result.
- **Introduction:** Establish the successful old view, expose its precise tension, derive the new view from one insight, preview the method and headline evidence, then state what changes for the field. Prefer a narrative over a generic contribution list.
- **Related work:** Group work by the distinctions needed for the central contrast. End each cluster with the paper's exact difference. Place this section early, integrate it into the introduction, or move it later according to argument flow—not habit.
- **Method:** Show the mental model first, then define the objective and notation. Describe a familiar baseline plus one exact change, formalize it, and explain why every component exists. Move recipes and exhaustive derivations out of the main flow.
- **Experiments and results:** Open with the evaluation contract. Give each subsection one claim-like question or takeaway. Order evidence as mechanism, decisive comparison, breadth or scale, and boundary when that matches the reader's objections. Report the controlled comparison and number before interpreting it.
- **Conclusion or discussion:** Extract the new mental model, the strongest open question or boundary, and the direction it unlocks. Do not replay the contribution list.
- **Appendix:** Preserve the main story's pace. Put reproducibility details, proofs, exhaustive ablations, full tables, qualitative sets, and failure cases here; the central claim and evidence must remain understandable without it.

If the author shares prose, diagnose the argument before rewriting it. Preserve technical meaning, citations, and the intended central claim. Never delete or weaken that claim merely because its evidence is incomplete; keep the bold claim in the draft and, only when material, flag the evidence gap in **Claim check**.

## Draft with corpus-level traits

- Organize the paper around one named conceptual primitive.
- Describe the method as a familiar baseline plus one exact change before adding formal detail.
- Make the method feel like a direct response to the diagnosed limitation.
- Use explicit contrasts and parallel lists only when they encode real structure.
- Prefer concrete nouns, active verbs, quantitative statements, and short transitions.
- Pair every important result with the claim it supports.
- Make each experiment deliver one sentence-level message: question, controlled contrast, observed result, and conclusion.
- Choose controls that favor a skeptical reader; state what is held fixed and disclose the strongest boundary.
- Prefer assertive verbs such as “shows,” “demonstrates,” and “establishes” when the stated experiment directly supports the claim. Reserve hedges for genuine extrapolation, not politeness.
- Favor simple, falsifiable explanations over ornamental novelty language.
- End with the scientific meaning, not a repeated contribution list.

For readability, keep one logical move per paragraph: topic sentence, necessary evidence or reasoning, then the implication. Introduce notation only after its purpose is intuitive, use one term for each concept, and delete details that do not change the reader's understanding.

Use figures as arguments. Default to a teaser result figure followed by an intuitive method figure. Make captions state the setup, control, metric, and takeaway so they can be understood without reading every surrounding sentence.

## Interaction contract

During discovery, respond compactly:

**Target story:** State the bold old-view → new-view → payoff arc in at most three bullets or three sentences. Surface only what changed or matters to the current decision.

**Decision now:** Exactly one question.

Keep the first coaching reply under roughly 120 words unless the author asks for a full outline or experiment plan. Do not surface every boundary or fill in unverified related work.

When enough information is locked, propose a short outline and ask for the next material choice before writing long prose. When the author says “write now,” draft from known facts, mark gaps, and continue without blocking. Keep any **Claim check** after the draft, outside the prose, and at no more than two short bullets. When asked to “polish only,” preserve the framing; put any serious mismatch outside the prose rather than weakening the draft.
