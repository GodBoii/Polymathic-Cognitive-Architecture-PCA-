# 02 - Neuroscience Principles

This file distills neuroscience into engineering principles for PCA. The goal is not perfect biological fidelity. The goal is to identify mechanisms that explain human-like flexibility, low energy use, robust perception, and lifelong learning.

## Principle 1: Specialized Systems With Different Learning Rules

Doya's computational framework maps major brain structures to different learning roles:

- cerebellum: supervised learning and internal/forward models;
- basal ganglia: reinforcement learning, value, and action selection;
- cerebral cortex: unsupervised/statistical representation learning.

Source: https://web.math.princeton.edu/~sswang/developmental-diaschisis-references/doya00_neural-networks.pdf

Engineering translation:

- Use multiple learning systems rather than one universal loss.
- Let different modules specialize in prediction, value, control, retrieval, and abstraction.
- Build a coordinator that assembles these systems for goal-directed behavior.

## Principle 2: Complementary Learning Systems

The hippocampus can rapidly encode new experience without overwriting slow cortical knowledge. The neocortex integrates structure gradually through repeated interleaved replay.

Source: https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf

Engineering translation:

- `episodic_memory`: append-only events, fast writes, high detail.
- `semantic_memory`: compressed concepts, slow updates, de-duplicated knowledge.
- `procedural_memory`: learned skills, policies, scripts, habits.
- `working_memory`: small active state used for current reasoning.
- `replay_engine`: reactivates selected episodes for consolidation and planning.

## Principle 3: Pattern Separation And Pattern Completion

The hippocampus helps keep similar experiences distinct, while also supporting recall from partial cues.

Sources:

- Pattern separation review: https://pmc.ncbi.nlm.nih.gov/articles/PMC3183227/
- Pattern completion/separation mechanisms: https://pmc.ncbi.nlm.nih.gov/articles/PMC3812781/

Engineering translation:

- Store episodes with discriminative context: time, place, goal, emotion/value, entities, sensory state.
- Use sparse keys to avoid merging similar memories too early.
- Retrieve by partial cue, then reconstruct missing context.
- Keep "this reminds me of X" separate from "this is X."

## Principle 4: Prediction As Multi-Modal Control

Predictive coding frames perception as top-down predictions plus bottom-up errors. Active inference adds action: agents act to sample the world in ways that reduce uncertainty and fulfill goals.

Sources:

- Predictive coding for AI: https://arxiv.org/html/2308.07870v3
- Active inference/free-energy overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC8871280/
- Canonical cortical microcircuits: https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/

Engineering translation:

- Maintain world state hypotheses.
- Predict next sensory state and reward under candidate actions.
- Use error signals to update perception and attention.
- Spend compute on surprising or goal-relevant signals.

## Principle 5: Global Workspace And Broadcast

Global workspace theory proposes that many specialized processes operate in parallel, while a limited-capacity workspace broadcasts selected information globally for reasoning, report, planning, and control.

Sources:

- Global workspace review: https://pmc.ncbi.nlm.nih.gov/articles/PMC2771980/
- GWT robotics review: https://link.springer.com/article/10.1007/s43154-021-00044-7
- LIDA cognitive cycle overview: https://en.wikipedia.org/wiki/LIDA_%28cognitive_architecture%29

Engineering translation:

- Many modules propose interpretations and actions.
- Attention selects a small set of contents.
- Selected content is broadcast to memory, planner, language, action selection, and learning modules.
- This gives the system a serial "thought stream" on top of parallel unconscious processing.

## Principle 6: Prefrontal Control And Working Memory

Prefrontal cortex and related networks support goal-directed behavior, cognitive control, task switching, working memory updating, and inhibition of automatic responses.

Source: https://www.nature.com/articles/s41386-021-01132-0

Engineering translation:

- Keep an explicit active task model: goal, constraints, current plan, uncertainty, stop conditions.
- Add inhibitory control: the agent should suppress fluent but irrelevant responses.
- Add task-switching and self-monitoring as first-class mechanisms.

## Principle 7: Basal-Ganglia-Like Action Selection

The basal ganglia are strongly associated with reinforcement learning, value evaluation, action selection, habit formation, and gating.

Sources:

- Neurocomputational basal ganglia review: https://pmc.ncbi.nlm.nih.gov/articles/PMC2762323/
- Reinforcement learning in basal ganglia loops: https://pmc.ncbi.nlm.nih.gov/articles/PMC4408000/
- Doya unified view: https://web.math.princeton.edu/~sswang/developmental-diaschisis-references/doya00_neural-networks.pdf

Engineering translation:

- Use a candidate-action generator plus a value/gating module.
- Learn when to deliberate and when to run a cached habit.
- Route actions through safety, goal alignment, and feasibility checks.
- Keep reward broader than scalar success: novelty, uncertainty reduction, social feedback, energy cost, task completion.

## Principle 8: Cerebellum-Like Fast Forward Models

The cerebellum is associated with supervised learning, timing, prediction, error correction, and fast skilled behavior.

Sources:

- Computational principles of supervised learning in the cerebellum: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056176/
- Prediction signals in cerebellum: https://pmc.ncbi.nlm.nih.gov/articles/PMC7105376/
- Doya unified view: https://web.math.princeton.edu/~sswang/developmental-diaschisis-references/doya00_neural-networks.pdf

Engineering translation:

- Learn compact forward models: action -> expected state change.
- Cache repeated reasoning as fast skills.
- Use timing and error correction for tool use, UI navigation, robotics, or simulated tasks.

## Principle 9: Sparse, Event-Driven, Local Computation

The brain does not activate everything at once. Sparse activation and local signaling help manage energy cost. BrainFacts notes that much energy goes into signaling and synapses, so minimizing unnecessary communication matters.

Source: https://www.brainfacts.org/brain-anatomy-and-function/anatomy/2019/how-much-energy-does-the-brain-use-020119

Engineering translation:

- Prefer event-driven module activation.
- Use routing before heavy inference.
- Cache local memories near the modules that use them.
- Track compute budget as part of cognition.

## Principle 10: Hierarchical Modularity

Brain networks show modular and hierarchically modular organization. Modules support local specialization; connector hubs support integration.

Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2010.00200/full

Engineering translation:

- Organize PCA as modules-within-modules.
- Separate local processing from global broadcast.
- Make explicit connector nodes: router, workspace, memory index, value system.

## Principle 11: Embodiment And Grounding

Human cognition develops through a body acting in the world. LLMs lack direct sensorimotor grounding, which contributes to brittleness in physical commonsense and spatial reasoning.

Sources:

- Embodied AI from LLMs to world models: https://arxiv.org/html/2509.20021v1
- Easy Problems That LLMs Get Wrong: https://arxiv.org/html/2405.19616v1
- Spatial reasoning benchmark: https://www.ijcai.org/proceedings/2024/0701.pdf

Engineering translation:

- Even without a robot, use simulated embodied tasks.
- Give the agent persistent objects, space, actions, consequences, and affordances.
- Evaluate by interaction success, not only answer text.

## Principle 12: Offline Replay And Sleep-Like Consolidation

Memory replay during sleep and rest supports consolidation, abstraction, and perhaps planning. The key engineering idea is that learning continues offline by replaying important or surprising episodes.

Sources:

- Hippocampal memory consolidation during sleep: https://pmc.ncbi.nlm.nih.gov/articles/PMC3117012/
- System consolidation during sleep: https://pmc.ncbi.nlm.nih.gov/articles/PMC3278619/
- MIT Picower replay overview: https://picower.mit.edu/discoveries/importance-memory-replay-and-sleep

Engineering translation:

- Add a background process that samples episodes.
- Consolidate into semantic memory and procedural skills.
- Rehearse rare but important failures.
- Use replay for counterfactual planning: "what if I had acted differently?"
