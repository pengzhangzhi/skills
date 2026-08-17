# Recent Section Style: 26 Kaiming He Coauthored Works

## Scope

This review covers every verified distinct work by the MIT vision researcher Kaiming He whose first public version or publication falls from **2023-08-16 through 2026-08-16**. Conference and arXiv versions are counted once; the renamed RCG paper and the arXiv/TMLR versions of DHN are each one work. The result is 26 works.

Use these as corpus-level patterns, not proof that one coauthor wrote a given sentence. All 26 inform coverage; repeated patterns across small-team technical papers carry more weight than a single large-consortium exception. Sources were cross-checked against the [official publication page](https://people.csail.mit.edu/kaiming/) and [DBLP record](https://dblp.org/pid/34/7659.html). DBLP's kidney-transplant entry was excluded after the primary article identified that Kaiming He as a different researcher at Guangdong Provincial People's Hospital.

## Section playbook

### Abstract

Use one of two compact arcs:

- **Belief challenge:** accepted belief → direct challenge → controlled finding → corrected view.
- **New paradigm:** important bottleneck → one named primitive → how it changes the formulation → decisive result → new direction.

Open with substance, not “recent advances.” Name the primitive early. Include one headline number when it makes the claim tangible. End on the scientific implication, often what the community can now reconsider or pursue. Do not spend abstract space on caveats unless they change the claim.

### Introduction

Build six moves in narrative form:

1. Establish why the successful existing paradigm matters.
2. Expose one precise mismatch, hidden assumption, or missing regime.
3. Ask the paper's real question, explicitly when useful.
4. Give the intuitive insight before machinery.
5. Introduce the named method as the natural consequence of that insight.
6. Preview the strongest evidence and the changed mental model.

Avoid the generic “three contributions” ending unless the method truly solves three parallel obstacles. Recent papers often state findings directly and let the paragraph structure carry the contribution logic.

### Related work

Most papers use a compact early section; discovery papers sometimes replace it with history/background, DyT places it late, and large generalist papers integrate it into the introduction. Choose placement for argument flow.

Organize by two or three conceptual axes that define the paper's contrast. Summarize a line of work, then end with **unlike / in contrast / orthogonal to / missing from** and the exact distinction. Do not write a citation chronology or claim novelty by adjective.

### Method

Start with the object being changed: distribution, representation, prediction target, operator, loss, or interface. Show the old and new views in one conceptual figure or equation. Then proceed:

1. minimal setup and notation;
2. one named primitive or identity;
3. a small number of components, each tied to a stated obstacle;
4. the training/inference consequence;
5. implementation details only if they affect understanding.

The common explanatory form is “familiar framework + one exact change.” Formalism earns its place by making the new view inevitable or testable, not by displaying complexity.

### Experiments and results

Open with the evaluation contract: data, metric, baselines, protocol, and what is held fixed. Organize subsections as answers. A strong default sequence is:

1. **Mechanism:** a toy study, controlled property test, or decisive ablation.
2. **Main result:** the fair comparison that delivers the headline.
3. **Breadth:** other models, datasets, tasks, scales, or resolutions.
4. **Boundary:** failure case, cost, weak regime, or unresolved alternative.

Swap the first two when the headline result is the hook. Use declarative headings and captions that state the result, not merely the topic. Pair every number with its interpretation, and isolate the one simple-looking detail that makes the method work. Strong controls—not cautious adjectives—earn direct claims.

### Conclusion or discussion

Keep the conclusion short unless the paper is explicitly opening a paradigm. Restate the learned principle, not the implementation or contribution list. Then name the most productive open question and the direction the result unlocks. A discussion may speculate boldly after clearly separating an observed result from an open theoretical claim.

### Appendix

Use the appendix to protect the main argument's pacing:

- implementation, hyperparameters, compute, and protocols;
- proofs, derivations, and algorithm details;
- exhaustive ablations and full comparison tables;
- additional tasks, resolutions, samples, editing cases, and visualizations;
- failure cases, limitations, and broader-impact material when venue conventions require them.

Cross-reference it at the point of need. Do not hide the experiment that establishes the central claim in the appendix.

### Opening figures

Across the corpus, the first two figures repeatedly serve two jobs even when their order varies:

- **Result hook:** the surprising win, visual capability, failure of the old view, or decisive scaling curve.
- **Method intuition:** the old/new contrast and the one primitive that changes the outcome.

For this skill, default to teaser result first and intuitive method figure second. Make both captions self-contained arguments with the setup, comparison, takeaway, and any curation or post-processing that affects interpretation.

## Corpus audit

Each entry records the dominant story and the most useful writing signal.

### 2023–2024

- [Return of Unconditional Generation / RCG](https://arxiv.org/abs/2312.03701) — New paradigm. The abstract turns a historical conditional/unconditional gap into one representation-conditioning primitive and a 64% relative FID reduction. The introduction pairs a method diagram with an immediate result figure; experiments use declarative “observations”; the appendix holds implementation, ablations, extra datasets, samples, and limitations.
- [Deconstructing Denoising Diffusion Models for Self-Supervised Learning](https://arxiv.org/abs/2401.14404) — Resurrected baseline plus discovery. The paper makes the research procedure itself the story: remove modern components one by one until a classical DAE remains. The main section is a sequence of controlled deconstructions; the conclusion states one critical component; the appendix is only recipe and transfer detail.
- [A Decade's Battle on Dataset Bias](https://arxiv.org/abs/2403.08632) — Discovery and countervailing wisdom. The abstract leads with a surprising 84.7% result; the introduction is historical tension; Figure 1 makes the reader play the experiment. “Brief history” replaces generic related work, result headings state observations, and a separate appendix records controls, extra results, and limitations.
- [Dynamic Inhomogeneous Quantum Resource Scheduling](https://arxiv.org/abs/2405.16380) — Problem solving. It enumerates the physical scheduling obstacles, maps them to a simulated environment and a Transformer RL agent, and closes with a numbered achievement list. This is a domain-engineering exception rather than a canonical prose model; its supplement carries extensive scientific background and simulator detail.
- [TetSphere Splatting](https://arxiv.org/abs/2405.20283) — New representation. The opening figure contrasts Eulerian and Lagrangian geometry; method sections separate the representation from initialization/optimization; experiments balance mesh quality with reconstruction accuracy. The long appendix carries metric formulas, geometric derivations, ablations, and extra results.
- [Physically Compatible 3D Object Modeling](https://arxiv.org/abs/2405.20510) — New formulation plus problem solving. The abstract names three orthogonal physical attributes; Figure 1 shows the visible failure of existing reconstructions before the pipeline. Evaluation ends in simulation and 3D-printing utility; limitations follow the conclusion; the appendix supplies gradients, printing, and simulation details.
- [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838) — Countervailing wisdom and new paradigm. The first sentence states the convention, then removes its supposed necessity with Diffusion Loss. The method moves from per-token probability to standard and masked autoregression; the discussion broadens the idea beyond images; supplementary sections hold recipes and extra comparisons.
- [Scaling Proprioceptive-Visual Learning with HPT](https://arxiv.org/abs/2409.20537) — Scaling and problem solving. “Heterogeneity” is the single roadblock; a modular trunk/interface design answers it. Pre-training and transfer get separate experiment sections, while the conclusion leads with real limits before returning to the perspective; failure cases live in the appendix.
- [Fluid](https://arxiv.org/abs/2410.13863) — Controlled discovery plus horse race. The paper studies a clean 2×2 design space—continuous/discrete tokens and random/raster order—before scaling the winning choice. Figure 1 is a qualitative result hook, with its special upscaling disclosed in the supplement; the supplement also records aligner ablations, prompt details, samples, failures, and guidance settings.

### 2025

- [Is Noise Conditioning Necessary?](https://arxiv.org/abs/2502.13129) — Direct belief challenge. The abstract literally states the accepted belief and challenges it; analysis precedes the proposed model, so the paper first explains the surprising graceful degradation and then exploits it. The discussion connects the finding to classical energy models and flows; the large appendix contains numerical protocols, theory, derivations, and samples.
- [Fractal Generative Models](https://arxiv.org/abs/2502.17437) — New paradigm. A familiar analogy—modularization and mathematical fractals—makes the recursive generator intuitive before formalization. The paper separates general framework from one image instantiation, uses pixel generation as a hard proof case, and ends by inviting a design space rather than replaying scores.
- [Denoising Hamiltonian Network](https://arxiv.org/abs/2503.07596) — Generalization and unification. Two limitations of local forward simulation motivate a denoising Hamiltonian operator; Figure 1 carries the thesis. Three distinct reasoning tasks test flexibility, while the discussion asks larger scientific questions and states compute/scale limits; derivations and physical interpretation move to the appendix.
- [Transformers without Normalization](https://arxiv.org/abs/2503.10622) — Countervailing wisdom with a tiny primitive. An abstract-like opener goes belief → observation → one-line DyT replacement → broad validation. Figure 1 makes the drop-in change instantly legible; related work is delayed until after experiments and analysis; limitations precede a crisp conclusion; the appendix is configuration-heavy.
- [Mean Flows](https://arxiv.org/abs/2505.13447) — New paradigm. “Average velocity” is contrasted with instantaneous velocity, derived as an identity, and converted into a self-contained one-step model. Figure 1 delivers the one-step result; the conclusion extracts a multiscale-simulation analogy; implementation, derivations, and samples stay in the appendix.
- [Highly Compressed Tokenizer Can Generate Without Training](https://arxiv.org/abs/2506.08257) — Discovery and resurrected baseline. The paper begins with crude token manipulations that unexpectedly edit semantics, then turns the finding into gradient-based test-time generation. Evaluation is organized by capability rather than leaderboard; the discussion reinterprets what the tokenizer itself has learned; appendices deepen the semantic probes and examples.
- [Diffuse and Disperse](https://arxiv.org/abs/2506.09027) — Simple primitive. A decade-long separation between generation and representation learning motivates one plug-and-play dispersive regularizer. The method emphasizes no pretraining, additional parameters, or external data; experiments repeat controlled gains across strong bases; the conclusion elevates “minimal interference” into a design principle.
- [Back to Basics](https://arxiv.org/abs/2511.13720) — Countervailing wisdom and paradigm reset. The title, abstract, and introduction all insist that denoising models should predict clean data. The manifold assumption provides the intuitive reason; theory and toy analysis precede scale results; the discussion turns the result into a minimalist, latent-free direction; recipes and exhaustive experiments move to appendices.
- [ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761) — Bold reframing. The title is the claim; the abstract contrasts the dominant language view with image-to-image translation, then gives the 60.4% hook. The method is deliberately ordinary once the representation changes; visual analysis supports the explanation; the conclusion asks the community to use ARC as a visual-generalization testbed.
- [Improved Mean Flows](https://arxiv.org/abs/2512.02012) — Parallel problem solving. The abstract names exactly two fastforward challenges and pairs each with one solution. Method subsections mirror that map, experiments verify stability/flexibility before the headline FID, and the short conclusion argues for stand-alone fastforward generation; implementation and samples are appended.
- [Bidirectional Normalizing Flow](https://arxiv.org/abs/2512.10953) — Constraint removal and paradigm expansion. The story revisits a classical framework, identifies exact analytic inversion as the bottleneck, and learns the reverse map instead. The conclusion uses a clean “from X to Y” sequence to state what changed; appendices hold method details, more experiments, editing, and visualizations.

### 2026

- [Pixel Mean Flows](https://arxiv.org/abs/2601.22158) — Synthesis around one missing regime. The abstract names two dominant traits—multi-step and latent-space generation—then separates network output space from loss space to remove both. Toy experiments precede ImageNet evidence; the conclusion extracts the end-to-end noise-to-pixels principle; implementation and visualizations are appended.
- [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) — Explicit new paradigm. A single distribution-pushforward picture contrasts iterative training-time evolution with diffusion's iterative inference. The method defines the drifting field, the experiments lead with one-step results, and the discussion states both the new mental model and the main unresolved theoretical implication.
- [GeoPT](https://arxiv.org/abs/2602.20399) — New pretraining paradigm. The geometry/physics gap is the bottleneck; synthetic dynamics “lift” cheap geometry into useful supervision. Experiments emphasize task breadth, data savings, convergence, and scaling; the extensive appendix carries theory, full investigations, showcases, recipes, tables, and limits.
- [Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329) — Discovery turned manifesto. The abstract states the paradigm shift openly; the introduction frames a falsifiable test of whether a generator is already a foundation model. Prior work is integrated rather than isolated, task-by-task evidence replaces a conventional Experiments section, the RGB-output interface unifies tasks, and a manifesto-like Discussion replaces a conclusion.
- [ELF](https://arxiv.org/abs/2605.10938) — Minimal interface change. The abstract contrasts discrete language with continuous image flows, then discretizes only at the final step. Figure 1 is the result hook and Figure 2 the method intuition; experiments stress quality, steps, and training tokens; the appendix contains a survey, distillation, method detail, ablations, protocols, and examples.
- [Video Generation Models are General-Purpose Vision Learners](https://arxiv.org/abs/2607.09024) — New pretraining paradigm. An NLP analogy produces three explicit requirements for vision—spatiotemporal structure, language alignment, and scale—then argues video generation satisfies all three. Figure 1 combines pipeline and paradigm shift, broad task results establish the claim, and the conclusion ends on unified physical-world intelligence. The primary arXiv version has no appendix, a useful reminder that supplements are optional when the main paper is self-contained.

## Fast drafting check

- Can the abstract be reduced to one surprise and one primitive?
- Does the introduction make the method feel inevitable?
- Does related work sharpen one contrast?
- Does the method explain before it formalizes?
- Does every experiment title state the question or finding?
- Do the first two figures deliver the result hook and method intuition?
- Does the conclusion change the reader's mental model?
- Does the appendix preserve completeness without carrying the central story?
