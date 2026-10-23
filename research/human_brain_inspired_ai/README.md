# Human-Brain-Inspired AI Research

Date: 2026-06-24

This folder is a curated research packet for shifting PCA away from "copy the LLM stack" and toward a brain-inspired cognitive architecture. It is not a claim that we can clone the brain directly. It is a design brief: what the human brain appears to do well, what current LLMs do poorly, and what architecture principles are worth turning into experiments.

## Core Thesis

The human brain should not be reduced to a next-token predictor. It is an embodied, energy-constrained, modular, recurrent, predictive, memory-rich control system. It learns across many time scales, mixes innate structure with experience, and constantly couples perception, memory, action, emotion, goals, and attention.

The strongest near-term path is not to simulate every neuron. It is to build a hybrid cognitive architecture with:

- specialized subsystems instead of one monolithic model;
- event-driven and sparse computation instead of full dense activation everywhere;
- fast episodic memory plus slow semantic consolidation;
- predictive world models tied to action, not only language;
- basal-ganglia-like action selection and reward learning;
- cerebellum-like fast skill models and forward prediction;
- cortical-style hierarchical representation learning;
- prefrontal/global-workspace-like working memory, attention, and task control;
- sleep/offline replay for memory integration;
- embodiment or at least simulated sensorimotor grounding.

## Important Corrections

The brain is not "a few million parameters." A common estimate is around 86 billion neurons and about 100 trillion synapses. The comparison with neural-network parameters is not one-to-one, because synapses are dynamic biological structures, neurons are active nonlinear cells, glia and neuromodulators matter, and the brain is embedded in a body.

The impressive point is still real: roughly 20 W of power supports perception, control, memory, planning, language, emotion, and robust real-world adaptation. That should push PCA toward energy-aware architecture, sparse activity, local memory, and active perception.

## Files

- [01_brain_facts_and_corrections.md](01_brain_facts_and_corrections.md) - factual grounding and corrections to common assumptions.
- [02_neuroscience_principles.md](02_neuroscience_principles.md) - brain systems and mechanisms that matter for AI design.
- [03_ai_architecture_implications.md](03_ai_architecture_implications.md) - proposed PCA direction and module blueprint.
- [04_research_roadmap.md](04_research_roadmap.md) - experiments, milestones, and evaluation metrics.
- [05_sources.md](05_sources.md) - annotated bibliography and source links.

## One-Sentence Direction

Build PCA as a small society of specialized learning systems with shared working memory, grounded world models, lifelong episodic memory, replay-based consolidation, and sparse event-driven computation.
