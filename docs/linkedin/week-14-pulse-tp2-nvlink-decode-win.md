# Splitting a model that fits on one card: what tensor parallelism actually buys you

Here's a setup that looks wasteful on paper. I took a 12B language model — quantized to 4-bit, so it sits comfortably inside a single RTX 3090's memory — and ran it across *two* GPUs instead of one. The model didn't need the second card. So what does splitting it actually buy?

The top-down answer is seductive and wrong: tensor parallelism splits the compute across two cards, so two cards should be twice as fast. Nobody who has done this expects a clean 2×, though. The interesting question was never *whether* a second card helps — it was **how much of that theoretical 2× survives once you pay the costs the napkin math ignores.** I expected a meaningful gain. I had no way to predict its size going in, because it hinges on a tug-of-war I couldn't resolve in my head.

## The tug-of-war

When you split a model with tensor parallelism (TP=2), you split its big weight matrices down the middle: each card holds half of each matrix and computes half of each matrix-multiply's result. Two things then pull in opposite directions:

- **In favor:** the slowest part of generating each token is streaming the model's weights through GPU memory. Halve the weights per card and the two cards read in parallel — that's a big bandwidth win, and token generation is bandwidth-bound, so it's exactly where it should help.
- **Against:** because each card only computed *half* of every result, the two cards have to stop and exchange their partial answers to reassemble the full one before they can continue. That exchange is called an **all-reduce**, and you pay it on every single token. How expensive it is depends entirely on how fast the link between the cards is.

A word on that exchange, because it's the crux. An all-reduce is a standard parallel-computing operation: every participant contributes a value, the values are combined (here, summed — each card's partial result plus the other's), and *every* participant gets the combined total back. The "all" is the important part — both cards need the full result, because both are about to compute the next step. The clever implementations arrange the cards in a ring and pass data around it in chunks, so no single card becomes a bottleneck and the bytes each card moves don't blow up as you add more cards — the bandwidth-optimal version of this trick came out of Baidu's deep-learning work and is what libraries like NVIDIA's NCCL run under the hood.

It doesn't happen after *every* matrix multiply, though — tensor parallelism is deliberately engineered to keep these exchanges rare. By splitting one matmul column-wise and the next one row-wise — the scheme NVIDIA introduced in its Megatron-LM work — each card can grind through a whole sequence of operations locally and only sync at the end. In practice that works out to two all-reduces per transformer block: one after the attention machinery, one after the feed-forward network. But "rare" is relative — a 12B model has dozens of blocks, and you pay those two syncs per block *for every token you generate*. On the NVLink bridge, where the exchange rides a fast (~100 GB/s) direct connection, each one is cheap relative to the compute it enables. On a slow link it would be ruinous.

How cheap, concretely? Each all-reduce shuffles a single hidden-state vector — for this model, about 7.5 KB of data, or roughly 15 KB once you count the round-trip across the link. The model has 48 blocks, so a full token costs ~96 of these exchanges: about **1.4 MB of traffic per token**. Now compare that to what the split *saves*: generating each token means reading the model's weights out of memory, and at TP=2 each card reads only its half — a saving measured in **gigabytes per token**. The thing tensor parallelism adds is on the order of a megabyte; the thing it removes is on the order of gigabytes. That's a ratio of roughly 2,000 to 1. Seen that way, the result stops looking surprising and starts looking inevitable — you're trading away gigabytes of memory reads for a megabyte of link chatter.

Which force wins — the halved-weight bandwidth saving, or the accumulated cost of all those syncs? That's an empirical question. So I measured it.

## The numbers

Same model, same prompt (8K tokens in, 512 out). Three configurations: one card (TP=1), two cards with tensor parallelism (TP=2), and two cards with *pipeline* parallelism (PP=2) — a different way to use both cards, included as a control.

```
                    1 card   TP=2 (NVLink)   PP=2 (NVLink)
  decode tok/s        70.0       102.6           72.2
  prefill tok/s     2,480       4,500          3,275

  under load (8 concurrent requests, aggregate tok/s)
                     112.4       193.3           117.0
```

Token generation got **47% faster** on two cards. Not the theoretical 2×, but a large, real gain — the halved-weights bandwidth win clearly beat the per-token all-reduce cost. The reason it's 47% and not 100% is the honest part: the all-reduce, the cache reads, and fixed per-step overhead *don't* halve, so they eat the gap to the ceiling. And under concurrent load the lead actually *grows* — two cards have twice the headroom to absorb a stacked-up queue before they saturate.

## Why the control matters

Look at the PP=2 column. Pipeline parallelism also "uses both cards" — but it resolves the tug-of-war differently, and on this workload it lands at roughly *single-card* performance. Same two GPUs, same NVLink bridge, nearly nothing gained.

That's the actual lesson. Using both cards isn't the variable — *which way you split the model* is. TP=2 won here because the bandwidth saving paid off and the all-reduce was cheap. Which leads to the one caveat worth stating plainly:

**This result is interconnect-dependent.** The whole thing rests on that all-reduce being cheap, which is true on NVLink. On a slow link, the per-token sync would dominate and the same split would lose. "TP=2 wins" isn't a law — it's the answer to *when* and *why*, and the why is the speed of the wire between your cards.

---

*Part of an ongoing series building a local LLM inference stack for an operator-copilot — incident investigation over logs, metrics, and live service components. The benchmark scripts and full result captures are in the public repo.*
