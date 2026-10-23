# 05 - Sources

This is an annotated source map for the research packet.

## Brain Scale, Energy, And Efficiency

### The Human Brain In Numbers

URL: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/neuro.09.031.2009/full

Use:

- Corrects the "100 billion neurons / 10x glia" myth.
- Gives the widely used 86 billion neurons and 85 billion non-neuronal cells estimate.
- Notes about 16 billion cortical neurons and 69 billion cerebellar neurons.
- Frames humans as a scaled-up primate brain, not a magical exception.

### UCLA Brain Research Institute: Billions Of Neurons, Trillions Of Synapses

URL: https://bri.ucla.edu/brain-fact/billions-of-neurons-trillions-of-synapses/

Use:

- Quick public summary: about 86 billion neurons and about 100 trillion synapses.

### BrainFacts: How Much Energy Does The Brain Use?

URL: https://www.brainfacts.org/brain-anatomy-and-function/anatomy/2019/how-much-energy-does-the-brain-use-020119

Use:

- Brain uses about 20 percent of resting body energy.
- Much energy goes to signaling and synapses.
- Task-related increases are small compared with baseline.
- Supports compute-budget-aware architecture.

### Brain Power

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8364152/

Use:

- Energy in cortex: communication is a major cost.
- Supports local/sparse computation as an architectural principle.

## Memory And Learning

### Why There Are Complementary Learning Systems

URL: https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf

Use:

- Hippocampus rapidly stores new items.
- Neocortex slowly integrates structure.
- Replay/reinstatement helps integrate new memories without catastrophic interference.

### Memory Consolidation

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4526749/

Use:

- Systems consolidation: memories initially dependent on hippocampus are reorganized over time.

### Hippocampal Memory Consolidation During Sleep

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3117012/

Use:

- Slow-wave/non-REM sleep is associated with gradual transfer of memories to neocortex.

### System Consolidation Of Memory During Sleep

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3278619/

Use:

- Declarative memory can be modeled as fast hippocampal store plus slow neocortical store.

### MIT Picower: Importance To Memory Of Replay And Sleep

URL: https://picower.mit.edu/discoveries/importance-memory-replay-and-sleep

Use:

- Accessible overview of replay during sleep and experience processing.

### Pattern Separation In The Hippocampus

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3183227/

Use:

- Motivates keeping similar episodes distinct.

### Pattern Completion And Pattern Separation Mechanisms

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3812781/

Use:

- Motivates recall from partial cues and separation of overlapping memories.

## Brain Modules And Learning Algorithms

### Doya: What Are The Computations Of Cerebellum, Basal Ganglia, And Cerebral Cortex?

URL: https://web.math.princeton.edu/~sswang/developmental-diaschisis-references/doya00_neural-networks.pdf

Use:

- Cerebellum as supervised/internal model learning.
- Basal ganglia as reinforcement learning/action selection.
- Cortex as unsupervised/statistical representation learning.
- Strong base for modular PCA design.

### Basal Ganglia Neurocomputational Models

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC2762323/

Use:

- Action selection, value learning, dopamine/RL models.

### Reinforcement Learning In Basal Ganglia Loops

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4408000/

Use:

- Dopamine and cortico-basal-ganglia-thalamo-cortical loops.

### Computational Principles Of Supervised Learning In The Cerebellum

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056176/

Use:

- Cerebellar architecture and supervised learning.

### Prediction Signals In The Cerebellum

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7105376/

Use:

- Cerebellum beyond classical motor error correction; predictive circuit function.

### Prefrontal Cortex In Cognitive Control And Executive Function

URL: https://www.nature.com/articles/s41386-021-01132-0

Use:

- Goal-directed behavior, cognitive control, working memory updating, inhibition, set shifting.

## Prediction, Attention, And Global Integration

### Predictive Coding For Brain-Inspired AI

URL: https://arxiv.org/html/2308.07870v3

Use:

- Predictive coding as asynchronous hierarchical inference and learning.

### Active Inference / Free Energy Principle

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8871280/

Use:

- Perception and action as inference under a generative model.
- Useful, but should be treated as one framework, not settled truth.

### Canonical Microcircuits For Predictive Coding

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/

Use:

- Maps predictive coding requirements to cortical microcircuit ideas.

### Global Workspace Model Of Consciousness

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC2771980/

Use:

- Parallel specialized processes plus limited global broadcast.

### Global Workspace Theory Review For Robotics

URL: https://link.springer.com/article/10.1007/s43154-021-00044-7

Use:

- GWT-inspired cognitive architectures for robots/consciousness modeling.

## Modularity And Connectome Organization

### Modular And Hierarchically Modular Organization Of Brain Networks

URL: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2010.00200/full

Use:

- Modules, hubs, connector nodes, robustness, adaptivity, evolvability.

### Small-World Brain Networks Revisited

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5603984/

Use:

- Brain networks combine local clustering with short global paths.

## LLM Limits, Grounding, And Embodiment

### Easy Problems That LLMs Get Wrong

URL: https://arxiv.org/html/2405.19616v1

Use:

- LLMs fail some simple logical, spatial, counting, linguistic, and relational tasks.

### Reframing Spatial Reasoning Evaluation In Language Models

URL: https://www.ijcai.org/proceedings/2024/0701.pdf

Use:

- Multi-hop spatial reasoning remains challenging.
- Real-world-inspired simulated spatial tasks are better evaluations than toy text relations.

### Large-Scale Evaluation Of Commonsense Knowledge In Humans And LLMs

URL: https://academic.oup.com/pnasnexus/article/5/3/pgag029/8487345

Use:

- LLMs remain below the human median in individual commonsense competence in the authors' evaluation.
- Human commonsense is heterogeneous and culturally variable.

### Embodied AI: From LLMs To World Models

URL: https://arxiv.org/html/2509.20021v1

Use:

- Embodied AI requires active perception, embodied cognition, and dynamic interaction.
- LLMs help semantic reasoning but are only part of an embodied system.
- World models support physical-law-compliant interaction.

### Zador: A Critique Of Pure Learning

URL: https://www.nature.com/articles/s41467-019-11786-6

Use:

- Animal brains are not blank-slate learners.
- Structured connectivity and innate priors enable rapid learning.
- Motivates architecture priors and developmental scaffolding.

## Cognitive Architectures

### ACT-R Overview

URL: https://act-r.psy.cmu.edu/about/

Use:

- Separates declarative memory, procedural memory, perceptual-motor modules, and buffers.

### Soar Cognitive Architecture

URL: https://arxiv.org/pdf/2205.03854

Use:

- Long-running cognitive architecture with working memory, procedural/semantic/episodic memory, chunking, and reinforcement learning.

### LIDA Cognitive Architecture

URL: https://ccrg.cs.memphis.edu/assets/papers/2013/franklin-ieee-tamd11.pdf

Use:

- Cognitive cycles, global workspace, action selection, emotion/motivation, and learning.
