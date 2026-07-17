# Coursework Taught Me to Talk About AI. This Is How I Learned to Do It.

I did the coursework. Two DeepLearning.AI specializations — Andrew Ng's Machine Learning, and
Mathematics for Machine Learning and Data Science — and three of Google's five-day Kaggle
intensives. They were real, I completed them honestly, and they worked at what they targeted.
The specializations taught me ML fundamentals. The intensives taught me parts of the
application layer: agents, prompt engineering, how to build on top of models.

And then I realized what none of it had touched: the layer in between. Transformers. GPUs.
Inference. I could hold a decent conversation about ML concepts and prompt engineering — but
tensor parallelism? Before this program, I had never even heard the term.

Some background, briefly. I took a year of mathematics and computer science at university in
Sweden straight out of high school — including linear algebra, which I then proceeded not to
use for decades, until machine learning made it unavoidable. I finished a degree years later,
the slow way, through evening and weekend classes. The career took the scenic route: some
years in fields that had nothing to do with technology, then into IT in the late nineties,
and by the early 2000s into internet infrastructure, where I've been ever since — sales
engineer and solutions architect on enterprise and carrier accounts, then technical product
management after relocating from Sweden to the US, and today a principal product architect.

So the gap I'd found was not a gap in ability to learn. It was a gap no amount of additional
coursework was going to close, because what was missing wasn't concepts — it was contact.
Real hardware, real failure modes, real measurements.

Last December I bought four used RTX 3090s, built them into a consumer-platform host, and
made a decision that shaped everything after it: the learning itself would be run as an
engineered system, with the same rigor I'd demand of a production project. The program ran
sixteen weeks, in phases, entirely on my personal time and personal equipment.

One more thing to establish, because it's load-bearing for everything below: this was not a
solo project, and I'm not going to write about it as if it were. The "we" in this article is
a partnership — me, the author, and Claude — Anthropic's AI — as tutor, scribe, and lab
partner. The judgment calls, the commits, and every word you're reading are mine to answer
for. The work was ours.

This article is about how that partnership was structured — because the structure, not the AI
alone, is what made a sixteen-week self-training program actually work.

## The backbone: a plan we could change without lying about it

The program ran on a
[training plan](https://github.com/mathiassamuelson/inference-engineering-training/blob/main/docs/training-plan.md)
we designed together before the first measurement: phases, weekly experiments, and the
sequence between them. The plan was never sacred. What was sacred was the record of changing
it: every amendment landed in the plan's own change log, dated, so the plan we finished with
never pretended to be the plan we started with.

That distinction — change freely, record honestly — turned out to be the program's entire
character in miniature.

## The rules of the game

Because here's the thing about being your own student, teacher, and lab director at once:
nobody checks your work unless you build the checking in. So we ran the program under a small
set of standing rules. Predict before measuring, in writing, then score the prediction. One
experiment at a time. When something we expected to matter turned out not to, that result
stayed in the record as a finding — not quietly dropped. Journals never rewritten —
corrections appended, dated, below the mistake. Don't build on a known bug. And what we
learned got written down with the same care as what we measured.

Each of those rules was forged by a specific moment in the program — a near-miss where two
wrong conclusions almost came out of one warm-up quirk, an improvement we were sure of that
measured out to no effect at all, a week where the program halted entirely rather than
measure on a known-broken stack.
The full method, and the moment that forged each rule, is
[a document in the repo](https://github.com/mathiassamuelson/inference-engineering-training/blob/main/docs/training-method.md).

One example: before the program's final characterization in Week 15, we wrote down five
predictions — one of which we committed to treating as proof that our explanation was wrong
if it failed. All five held, and the scorecard still notes the places where the observations
drifted from what we'd predicted. Checking your work on the wins, not just the misses, turned
out to matter more than I expected.

And all of it is public. The training plan, the daily journals, the results, the compendium,
the claim inventories, and the documents built from them live in
[a public repository](https://github.com/mathiassamuelson/inference-engineering-training)
that anyone can audit — including every appended correction. Rules you enforce in private are
suggestions. The repo is what made these rules real.

## One subject per session

Now the partnership mechanics — the part I most wish someone had written down for me in
December.

Every working session had exactly one subject. What I found is that keeping AI sessions
narrowly focused all but eliminates the risk that adjacent information drifts into the
context and quietly influences outcomes — context rot, to give it its name. It is the same
hygiene you'd apply to an experiment: control what's in the flask.

And a focused session ends with a property worth exploiting: at that moment, the context is
fully wired with exactly what the next session needs. So at each session's close, Claude
wrote the next session's opening prompt itself — scope, what's explicitly out of scope, the
working contract, and the pre-session checks to run before touching anything. A deliberate
compaction of working context into a handoff. Each fresh session opened from that prompt,
inside a Claude Project — the workspace feature whose shared knowledge base held the training
plan — so every session started narrow but never lost sight of the big picture.

## The right surface for the work

Work was routed by type. Design, prediction, and prose reasoning ran in the chat interface —
dialogue-shaped work. Command-heavy, mechanistic execution — running measurements, applying
edits across files, git operations — ran in Claude Code. From Week 15 on, the record shows
this as an explicit convention: each session journal opens by pairing its genre with its
tool. That's when the convention got its name in the record; the split itself had taken shape
gradually over the weeks before, and I honestly can't point to the moment it emerged.

## Governed like everything else

The partnership itself ran under the program's rules — that's the part that made it more than
autocomplete with enthusiasm. Like every rule in the program, the strictest one was added
when the work showed the need — it just happened to be the last to arrive. For the program's
final documents, the capstone first and now this article, nothing was drafted until a claim
inventory — every claim the document would make, tagged by evidential basis, with sources —
had been reviewed by me and committed to the repo. The
[inventory behind the article you're reading](https://github.com/mathiassamuelson/inference-engineering-training/blob/main/phase-3-optimization-and-quantization/week-16/week-16-session4-method-pulse-claim-inventory.md)
is in the public record. Claude's drafts arrived carrying inline evidential tags; my review
pass challenged them against the record; the tags came off only at final.

The through-line: Claude wasn't used as an oracle. It was a partner bound by the same
contract as everything else in the program — its claims reviewable, its drafts correctable,
its output constrained by the same provenance rules that governed mine.

## A tutor with unreasonable patience

One more role deserves its own story. Linear algebra and I had an adversarial relationship
going back to that first university year. Midway through the sixteen weeks, we did a deep
decomposition of the transformer feed-forward network — from raw matrix mechanics up through
why the parallelization strategy we'd measured behaves the way it does — and committed it to
the repo as
[the first entry of a reference series](https://github.com/mathiassamuelson/inference-engineering-training/blob/main/docs/compendiums/entry-01-ffn-inference-shapes-tensor-parallelism.md).
Deriving why the architecture is shaped the way it is, rather than memorizing that it is, is
what finally defused a thirty-five-year grudge against linear algebra. Just as well — there's
no avoiding it in this field. Linear algebra is simply part of my life now, whether I like it
or not. These days, mostly, I do.

That is what the tutor role looked like in practice: not handing out answers, but walking me
through the why — at whatever depth the question deserved, for as long as it took.

## What it ran on, and thanks owed

The hardware deserves a mention for what it says about accessibility: four used, six-year-old
consumer NVIDIA GPUs carried this entire program — including multi-GPU serving over NVLink —
on a desktop platform. The frontier of this field is expensive. The on-ramp is not pocket
change either — but it is within an individual's reach. What
surprised me more: the hardware's age was never the obstacle I expected it to be. Current
serving frameworks and current models still support these cards, and where their age did
show — a few newer capabilities the silicon simply lacks — ready workarounds existed and
worked remarkably well.

And the honest thanks: I could not have learned this alone. What Anthropic's Claude provided
wasn't answers — it was a design partner available at the depth and cadence a program like
this demands, and one that could be bound: held to the prediction contracts, the review
stops, and the provenance rules like any other part of the system.

## Where this goes

I set out hoping to move from knowing things about this field toward reasoning about it from
first principles. I won't claim to have arrived — the record shows the practice, not the
destination. But the direction of travel is real. A certificate says I completed someone
else's curriculum. The paper trail shows the work itself — the predictions, the misses, the
corrections — and anyone can walk through it. That difference is the distance traveled.

The four GPUs are being retired and the hardware upgraded, and the next level of this
training will run on the new platform — the same partnership, the same method. The
measurements retire with the machine. The method — and the way of working that produced it —
is what transfers.

That was the point all along.
