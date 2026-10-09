### Qwen3TTS

```bash
========================================================================================================
                                     Qwen3-TTS TRAINING FLOW (Speaker ID)
========================================================================================================

  [ Text Data ]             [ MetaData ]                              [ Target Audio Data ]
  ("نص بصوت عمر")         (speaker: "omar" -> ID: 105)                 (صوت عمر الحقيقي)
        │                          │                                           │
        ▼                          ▼                                           ▼
 [ Text Tokenizer ]       [ Speaker Embedding Table ]             [ Qwen-TTS-Tokenizer-12Hz ]
        │                 (Trainable Matrix: N_spks x D)                       │
        ▼                          │                                           ▼
 { Text Tokens }                   ▼                                 { Target Codec Tokens }
 [t1, t2, t3...]          { Speaker Vector: S_omar }                 [C0, C1, ..., C15] (Target)
        │                          │                                           │
        └──────────────────────────┴───────────────────────────────────────────┤
                                   │ (Concatenation / Injection)               │
                                   ▼                                           │
                       ┌──────────────────────┐                                │
                       │ Qwen3 LM (Backbone)  │                                │
                       └──────────────────────┘                                │
                                   │                                           │
                                   ▼                                           │
                        { Hidden State: H_t }                                  │
                                   │                                           │
                                   ▼                                           │
                       ┌──────────────────────┐                                │
                       │     MTP Module       │                                │
                       └──────────────────────┘                                │
                                   │                                           │
                                   ▼                                           ▼
                      { Predicted Codec Tokens } ────────────────► [ Loss Computation ]
                   [C0_pred, C1_pred, ..., C15_pred]               (Cross-Entropy Loss)
                                                                               │
                                                                               ▼
                                                                 [ Backpropagation / Gradient ]
                                                                 - Updates Qwen3 LM & MTP
                                                                 - Updates S_omar in Embedding Table
========================================================================================================
```