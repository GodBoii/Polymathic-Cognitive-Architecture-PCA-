# Normal Decoder-Only LLM Architecture

```mermaid
flowchart TB
    A[User Prompt Text] --> B[Tokenizer]
    B --> C[Token IDs]
    C --> D[Token Embedding + Positional Encoding]
    D --> E["N x Transformer Decoder Block<br/>(Causal Self-Attn + FFN + Residual + RMSNorm/LayerNorm)"]
    E --> F[Final Norm]
    F --> G[LM Head Linear Projection]
    G --> H[Softmax over Vocabulary]
    H --> I[Sampling/Greedy Next Token]
    I --> J[Append Token to Context]
    J --> E

    K[Shifted Target Tokens] --> L[Cross-Entropy Loss]
    G --> L
    L --> M[Backpropagation]
    M --> N[Optimizer Step]
```

# PCA Three-Phase Architecture (Current Repo Implementation)

```mermaid
flowchart TB
    A[Input IDs] --> B[Embedding + Position Embedding]
    B --> C["Encoder Stack<br/>(EncoderBlock x encoder_layers)"]

    subgraph LW[Latent Workspace / Imagination]
        C --> D[Latent Seed]
        C --> E[token_to_latent cross-attention]
        D --> E
        E --> F["Iterative Latent Refinement x imagination_steps<br/>latent self-attn + latent FFN + anchor mix"]
        F --> R["Optional RMS Alignment<br/>toward imagination_target_rms"]
        R --> G[Latent Memory]
    end

    subgraph DEC[Decoder / Rendering]
        H[Decoder Input IDs] --> I[Embedding + Position Embedding]
        I --> J["Decoder Stack<br/>(causal self-attn + cross-attn to Latent Memory + FFN)"]
        G --> J
        J --> K[Final RMSNorm]
        K --> L[LM Head]
        L --> M[Logits]
    end

    M --> N[Cross-Entropy Main Loss]
    F --> O["Imagination Aux Losses<br/>(stability + consistency + norm + contraction)"]
    N --> P[Total Loss]
    O --> P
    P --> Q[Backprop]
    Q --> S[Optimizer (current local three-phase presets use Muon)]
```