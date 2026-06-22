# Did 4-bit quantization quietly make my models worse? Here's how I actually checked.

When you quantize a model to 4-bit, you do it for a reason — it fits on smaller hardware, it
serves cheaper, it runs faster. But you're trading something for that, and the honest question is:
*how much?* Most of the time the answer is a shrug and a few eyeballed outputs that "look fine."

I wanted better than a shrug. This post is the method I used to put a number on it — a repeatable,
position-bias-controlled LLM-as-judge evaluation that compares a quantized model against the
full-precision parent it came from, and tells you whether you actually gave anything up. I ran it
across two model tiers. The short version: at 4-bit, both models came out **quality-equivalent to
their full-precision parents** — and the method for showing that is what I want to walk through.

## What was being evaluated

The system under test is an **RCA-style operator copilot**: an assistant that reasons over a
platform's architecture, then investigates incidents by reading logs and metrics, issuing read-only
SQL, and respecting operational guardrails (never restart a service without confirmation, stay
read-only by default).

It's built as two tiers, and the tiering is the reason there are two models to evaluate:

- An **orchestrator** — @Google DeepMind **Gemma 4 31B** — handles open-ended, whole-incident
  reasoning.
- It fans work out to **worker sub-agents** — **Gemma 4 12B**, a smaller and faster model —
  that each do one focused job: pull the signals for a single component and hand back a structured
  result.

Sizing each tier to its job — the smaller model for focused, single-component work, the larger for
open-ended whole-incident reasoning — is a cost-and-throughput decision, not the subject of this
post. What matters here is that *each* tier runs as a 4-bit **QAT** checkpoint (W4A16,
compressed-tensors), and each QAT model was derived from a full-precision **BF16** version — the
parent. That gives two parent-and-quantized pairs, which is exactly what you want to compare.

A word on QAT, because it's why the result is even plausible. QAT — quantization-aware training —
stores the **weights** at 4 bits while keeping **activations** at 16 bits, and crucially it
*simulates* the 4-bit rounding during a fine-tuning pass, so the model learns to be good despite the
precision it'll run at. That's different from rounding a finished model's weights down to 4 bits
after training (post-training quantization), where the model never gets a chance to adjust for the
precision loss. The promise is "low precision without the quality hit" — and a comparison against the
parent
is how you check the promise actually held for *your* workload.

## The method

The whole evaluation rests on one idea: **don't diff the text, judge the substance — and control
for the judge's own biases.**

**1. Matched-provenance captures.** Each model answers the same fixed set of probes, under the same
system prompt, at greedy sampling (temperature 0). The system prompt's hash and the git commit are
recorded in every result file, so I can prove both models saw byte-identical inputs. The only thing
that differs between two runs is the model.

**2. Self-contained probes.** Each probe is a complete incident scenario with its evidence embedded
— a symptom, a log snippet, a metrics readout — so the model reasons from the evidence plus its
architectural knowledge. No live system moves underneath the test, which makes it deterministic and
reproducible.

**3. A structured rubric, scored by an LLM judge.** A strong model (@Anthropic Claude Opus) scores responses
against explicit, task-appropriate axes rather than a vague "which is better." The axes differ by
tier because the *jobs* differ — the orchestrator is judged on diagnostic accuracy, tooling choice,
next-action soundness, guardrail adherence, and clarity; the workers on extraction fidelity, scope
discipline, and signal correctness.

**4. Position-bias control.** LLM judges have a presentation-order bias:
show them response A then B and they lean one way; swap the order and the verdict can flip. So every
pairwise comparison is run **both ways**, and any verdict that reverses under the swap is recorded as
*order-sensitive* and collapses to a tie. This is what turns "the judge picked a winner" into
"the judge picked a winner *and it survived having its bias controlled for*." Skip the control, and a
verdict that's really just presentation order gets counted as a genuine quality difference.

**5. Pointwise scoring for the smaller model.** Pairwise tells you if QAT matches its parent. For the
smaller worker model there's a second, harder question — is it *good enough in absolute terms* for
the focused job, regardless of the parent? So the workers also get scored pointwise, 1–5 per axis.

**6. Format conformance checked deterministically, not by the judge.** The workers emit a strict JSON
contract. Whether output is valid JSON is a parser question, not a judgment call — and the judge is
explicitly told to ignore formatting — so that axis is a separate deterministic check, never mixed
into the LLM scoring. Right instrument for each question.

## What it found

**Orchestrator tier (Gemma 4 31B), pairwise across five axes:**

```
overall:   BF16 wins 2 | QAT wins 2 | tie 4   (3 of the ties were order-sensitive)

axis                    BF16  QAT  tie   order-sensitive
----------------------  ----  ---  ----  ---------------
diagnostic_accuracy       2    1    5          1
evidence_and_tooling      1    1    6          3
next_action_soundness     1    3    4          1
guardrail_adherence       0    0    8          1
communication_clarity     0    2    6          2
```

No axis shows a one-sided regression. The decisive verdicts that survived order-swapping split
**exactly evenly, two to each model** — so neither has a real edge. The standout is
`guardrail_adherence`: **8/8 tie.** On the axis that matters most for an operator copilot — refusing
to restart a service without confirmation, staying read-only — the 4-bit model is *indistinguishable*
from its full-precision parent. The only directional lean anywhere is a slight one toward QAT on
next-action soundness, all at low-to-medium confidence.

**Worker tier (Gemma 4 12B), two components, pairwise + pointwise + format:**

```
measure                  component A            component B
-----------------------  --------------------   --------------------
format (strict JSON)     6/6 QAT · 6/6 BF16     6/6 QAT · 6/6 BF16
pairwise (QAT vs BF16)   parity                 parity
pointwise overall (1-5)  4.83 QAT · 4.83 BF16   5.0 QAT · 5.0 BF16
```

On format, both models were perfect — every response was valid contract JSON, with nothing to
separate the quantized model from its parent. Pairwise is parity once order-sensitivity is
discounted. And pointwise — the absolute "is the small model good enough?" question — comes back
**yes**: both the quantized model and its parent land at 4.83 and 5.0 overall, with the hardest
reasoning probe (distinguishing a problem caused by an *outbound dependency* from one *internal* to
the service) scored clean on both.

### Two honest wrinkles

A method is only worth sharing if it surfaces its own warts:

- **Order-sensitivity is real and worth measuring.** Several "wins" reversed when the presentation
  order flipped. That's not a failure of the method — it's the method working. Those margins were
  thinner than the judge's own bias, which is itself the finding: the models are that close.
- **The small model copied the prompt's formatting.** On the first worker run, the model produced
  perfectly valid JSON *wrapped in a markdown code fence* — because the prompt's schema example was
  shown in a fence. It was mimicking the example's surface form, not failing to understand the
  contract. The fix was to mark the example as illustrative-only and demand a raw object; conformance
  went from 1-of-6 to 6-of-6. Smaller models imitate the *shape* of your examples — a useful thing to
  know before you trust one in production.

## What the quantization actually bought

Quality held. Here's the other side of the trade — weights on disk, the hardware-independent number:

```
model / format               weights on disk
---------------------------  ---------------
Gemma 4 31B  BF16 (parent)       59 GB
Gemma 4 31B  QAT  (W4A16)        22 GB
Gemma 4 12B  BF16 (parent)       23 GB
Gemma 4 12B  QAT  (W4A16)       9.6 GB
```

Roughly **2.7× smaller** at the larger tier and **2.4× smaller** at the smaller one — at no
measurable quality cost. Operationally that's the difference between a stack that fits and one that
doesn't: the quantized smaller model serves on a single 24 GB card where the parent needs two. The
full-precision 31B parent only loads at all by spreading across all four cards with pipeline
parallelism — which is how it was captured for this evaluation — but that monopolizes the whole box
and runs at latency I'd call non-interactive, so it isn't a way you'd actually serve the orchestrator.
The quantized 31B, by contrast, serves on a two-card pair and leaves the other two cards free for the
worker tier. (Throughput is its own story — a separate post.)

## The cost of knowing

The entire orchestrator-tier verdict — eight probes, scored both directions, across five axes —
cost about **$0.22** in judge API calls. That's the whole point of building it as a harness rather
than a vibe check: at that price, re-running with a revised rubric, more probes, or a second judge
model is never a budget question. The expensive part was deciding *what* to measure and *how to
control the judge* — and once that's built, it's reusable across every model you ship.

## What to take from this

- **Judge substance, not text.** Two equally-good answers worded differently will fool any
  text-diff. An LLM judge with an explicit rubric measures the thing you care about.
- **Always run both presentation orders.** Uncontrolled, an LLM judge's position bias manufactures
  fake winners. Order-sensitivity isn't noise to suppress — it's a signal that the candidates are
  close.
- **Match the instrument to the question.** Pairwise for "did it regress," pointwise for "is it good
  enough," and a deterministic parser for "is the format valid." Don't ask the LLM judge to measure
  formatting it's been told to ignore.
- **QAT is worth trusting — but verify on your workload.** Quantization-aware training really can be
  near-lossless. "Really can be" is not "is" until you've checked it against the parent on the task
  you actually run. The check is cheap. Run it.
