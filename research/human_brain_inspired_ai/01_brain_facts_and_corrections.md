# 01 - Brain Facts And Corrections

## 1. The Brain Is Not Small In Raw Capacity

The human brain is often described as having about 86 billion neurons and about 85 billion non-neuronal cells. Herculano-Houzel's review reports that Azevedo et al. found an adult male human brain of about 1.5 kg to contain about 86 billion neurons and 85 billion non-neuronal cells, with about 16 billion neurons in the cerebral cortex and about 69 billion in the cerebellum.

Source: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/neuro.09.031.2009/full

UCLA Brain Research Institute gives the common public-facing summary: about 86 billion neurons and an estimated 100 trillion synapses.

Source: https://bri.ucla.edu/brain-fact/billions-of-neurons-trillions-of-synapses/

Design correction:

- Do not describe the brain as "million parameter."
- Do not compare synapses directly to model parameters.
- Do compare the brain and LLMs by functional organization, learning efficiency, energy use, robustness, embodiment, and memory behavior.

## 2. The Brain Is Extremely Energy Efficient

BrainFacts/Society for Neuroscience reports that the adult brain consumes about 20 percent of the body's resting energy, and much of that energy goes to neural signaling, especially synapses. It also notes that the brain's task-related energy increase is usually small compared with its large baseline.

Source: https://www.brainfacts.org/brain-anatomy-and-function/anatomy/2019/how-much-energy-does-the-brain-use-020119

Design implication:

- Energy efficiency is not a side optimization. It is architectural.
- Sparse firing, local communication, myelination, prediction, and modularity reduce needless computation.
- PCA should treat compute as a scarce resource and route work only to relevant subsystems.

## 3. The Brain Is Not Just Prediction, But It Does Predict

The claim "the brain is not a next-token predictor" is right in spirit. The brain does not merely predict the next word. It predicts sensory input, body state, action consequences, rewards, social reactions, pain, balance, future trajectories, and hidden causes of the world.

Predictive coding and active inference theories frame the brain as using hierarchical generative models, prediction errors, and action to reduce uncertainty. But these theories are broader than language modeling: prediction is coupled to perception and action.

Sources:

- Predictive coding review: https://arxiv.org/html/2308.07870v3
- Active inference/free-energy overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC8871280/
- Canonical microcircuits for predictive coding: https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/

Design implication:

- PCA should not predict "next token" as the central objective.
- It should predict multi-modal state transitions: "what will I sense if I act?"
- Prediction should support action, memory, attention, and self-correction.

## 4. Human Learning Is Not Pure Blank-Slate Training

Zador argues that animal learning depends heavily on structured connectivity and innate behavioral priors, compressed through evolution into the genome. Young animals learn rapidly partly because they are not initialized as generic blank models.

Source: https://www.nature.com/articles/s41467-019-11786-6

Design implication:

- Do not rely only on scale and data.
- Build strong priors: objects, space, causality, agents, goals, affordances, time, uncertainty, and body-like constraints.
- Use "developmental scaffolding": start with simple sensorimotor skills and progressively integrate language and abstraction.

## 5. Human Memory Is Multi-System, Not One Context Window

The complementary learning systems theory says the hippocampal system rapidly stores new episodic memories, while the neocortex slowly integrates structure across repeated/interleaved experience. McClelland, McNaughton, and O'Reilly describe rapid hippocampal learning plus gradual neocortical learning as a way to avoid catastrophic interference.

Source: https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf

Sleep and replay appear important for consolidating memories and integrating experience.

Sources:

- Hippocampal memory consolidation during sleep: https://pmc.ncbi.nlm.nih.gov/articles/PMC3117012/
- System consolidation of memory during sleep: https://pmc.ncbi.nlm.nih.gov/articles/PMC3278619/
- MIT Picower memory replay overview: https://picower.mit.edu/discoveries/importance-memory-replay-and-sleep

Design implication:

- PCA needs persistent episodic memory, semantic memory, procedural memory, and working memory.
- It needs offline replay and consolidation, not just retrieval-augmented prompting.
- It needs mechanisms for pattern separation and pattern completion.

## 6. "Easy For A Child" Tasks Are Not Actually Easy

Moravec's paradox captures the idea that tasks humans find effortless, like perception, movement, object interaction, and commonsense physical reasoning, are often hard for AI. Modern LLMs can perform expert-looking symbolic tasks while still failing brittle commonsense, spatial, counting, and relational tasks.

Sources:

- Moravec's paradox background: https://en.wikipedia.org/wiki/Moravec%27s_paradox
- Easy Problems That LLMs Get Wrong: https://arxiv.org/html/2405.19616v1
- Spatial reasoning benchmark: https://www.ijcai.org/proceedings/2024/0701.pdf
- Commonsense evaluation in humans and LLMs: https://academic.oup.com/pnasnexus/article/5/3/pgag029/8487345

Design implication:

- Coding and math skill should not be mistaken for general intelligence.
- Evaluation should include physical commonsense, object permanence, spatial reasoning, causal interventions, and adaptive control.
- The system should be judged by reliability in simple grounded tasks, not only benchmark scores.

## 7. The Brain Is Modular And Integrated

Brain networks are modular and hierarchically modular: densely connected local modules, sparse long-range integration, and hub/connector nodes. This supports robustness, adaptivity, and evolvability.

Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2010.00200/full

Design implication:

- PCA should not be a single giant undifferentiated model.
- It should use specialized modules with controlled communication.
- The hard part is not only module quality, but routing, broadcast, and synchronization.

## 8. A Brain-Like AI Should Be Functional, Not Anatomically Literal

We do not need to simulate all neurons and chemistry to learn from the brain. A practical architecture should imitate computational roles:

- cortex-like representation learning;
- hippocampus-like fast memory;
- basal-ganglia-like action selection and reward learning;
- cerebellum-like forward models and skill automation;
- prefrontal-like working memory and task control;
- thalamus/salience-like routing and gating;
- global workspace-like broadcast for conscious-like integration;
- sleep-like offline replay and consolidation.

This is closer to cognitive architecture than to LLM scaling.
