# 04 - Research Roadmap

This roadmap turns the research into staged work. It assumes we want a working architecture, not only a theory document.

## Phase 0: Define The Target

Goal:

- Stop optimizing for "small LLM clone."
- Define PCA as an embodied cognitive architecture.

Deliverables:

- architecture spec;
- memory schema;
- cognitive cycle;
- evaluation battery;
- module interface contracts.

Key decision:

- Language is an interface, not the core substrate.

## Phase 1: Build A Cognitive Kernel

Implement:

- working situation model;
- episodic memory;
- semantic memory;
- action proposal interface;
- value/action gate;
- event log;
- replay job.

Avoid at this phase:

- training a large model;
- imitating transformer internals;
- overfitting to text-only benchmarks.

Success criteria:

- the agent can remember concrete episodes;
- it can retrieve similar prior situations;
- it can summarize repeated patterns into semantic memory;
- it can choose between candidate actions using goals and constraints.

## Phase 2: Add Simulated Embodiment

Implement a small world:

- grid or simple 2D object world;
- objects with containment, support, movement, and affordances;
- simple visual or symbolic observations;
- actions with consequences.

Tasks:

- find hidden object;
- move container;
- use tool;
- predict where an object will be after movement;
- recover after obstacle or changed goal.

Success criteria:

- performance improves after few episodes;
- learned knowledge transfers to new object names/layouts;
- failures are stored and avoided later;
- explanations refer to state and memory, not generic text.

## Phase 3: Complementary Learning System

Implement:

- fast episodic writes;
- slow semantic consolidation;
- replay prioritization;
- contradiction/confidence handling;
- memory decay and strengthening.

Replay priorities:

- surprise;
- reward;
- failure;
- user correction;
- novelty;
- high future usefulness.

Success criteria:

- reduced catastrophic forgetting;
- stable semantic rules emerge from episodes;
- rare but important failures remain retrievable.

## Phase 4: Action Selection And Habit Formation

Implement:

- candidate action generator;
- value model;
- procedural memory;
- chunking successful deliberations into procedures;
- habit inhibition when context changes.

Success criteria:

- repeated tasks become faster;
- old habits can be overridden by new constraints;
- action choices include uncertainty and risk.

## Phase 5: Predictive World Models

Implement:

- transition predictor;
- causal relation store;
- counterfactual simulator;
- active perception actions to reduce uncertainty.

Success criteria:

- agent can simulate likely outcomes before acting;
- agent asks for missing observations only when needed;
- agent improves from prediction error.

## Phase 6: Energy-Aware Sparse Routing

Implement:

- salience gate;
- module activation budget;
- compute accounting;
- local caching;
- interrupt handling for surprising events.

Success criteria:

- fewer modules activate on routine tasks;
- more modules activate on novel or risky tasks;
- compute decreases with skill acquisition;
- performance does not collapse under sparse routing.

## Phase 7: Language Integration

Only after the above, add language as:

- instruction parser;
- explanation generator;
- symbolic reasoning helper;
- communication layer;
- teacher feedback channel.

Success criteria:

- agent can ground words in memories, actions, and world state;
- language improves control, not just output fluency;
- agent can say "I do not know" when memory/world model is insufficient.

## Evaluation Battery

### Brain-Inspired Cognitive Metrics

- `sample_efficiency`: learns from few experiences.
- `memory_retention`: recalls relevant old episodes.
- `transfer`: applies knowledge to new layouts/tasks.
- `grounding`: ties answers to simulated or real observations.
- `intervention_success`: acts correctly, not only explains.
- `prediction_accuracy`: forecasts consequences of actions.
- `compute_efficiency`: solves routine tasks with less compute over time.
- `habit_flexibility`: overrides cached behavior when context changes.
- `self_correction`: updates after error or feedback.

### Anti-LLM-Brittleness Tests

- object permanence;
- multi-hop spatial relations;
- containment and support;
- simple counting under rearrangement;
- tool affordances;
- delayed goal resumption;
- ambiguous observation handling;
- contradiction detection.

## Reading Order

1. Brain facts and energy:
   - https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/neuro.09.031.2009/full
   - https://www.brainfacts.org/brain-anatomy-and-function/anatomy/2019/how-much-energy-does-the-brain-use-020119

2. Memory:
   - https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC3278619/

3. Learning systems:
   - https://web.math.princeton.edu/~sswang/developmental-diaschisis-references/doya00_neural-networks.pdf
   - https://www.nature.com/articles/s41467-019-11786-6

4. Modular networks and workspace:
   - https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2010.00200/full
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC2771980/

5. LLM limitations and embodiment:
   - https://arxiv.org/html/2405.19616v1
   - https://www.ijcai.org/proceedings/2024/0701.pdf
   - https://arxiv.org/html/2509.20021v1

## Near-Term Build Recommendation

Start with the cognitive kernel and simulated toddler-task world. This is more valuable than trying to train a neural model immediately, because it forces the architecture to handle memory, state, action, causality, and learning loops.

Concrete first milestone:

- A console demo where PCA sees a tiny object world, acts in it, stores episodes, consolidates rules, and performs better on later tasks.
