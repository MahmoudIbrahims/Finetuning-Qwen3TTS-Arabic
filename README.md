## Qwen3TTS



### Training Flow:

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

---------------------------------
### Inference Flow:

```bash
========================================================================================================
                                    Qwen3-TTS INFERENCE FLOW (Speaker ID)
========================================================================================================

    [ Text Input ]                                            [ Speaker ID ]
   ("أهلاً بك، أنا عمر")                                      (speaker: "omar")
          │                                                          │
          ▼                                                          ▼
   [ Text Tokenizer ]                                    [ Speaker Embedding Table ]
          │                                              (Loaded from checkpoint)
          ▼                                                          │
   { Text Tokens }                                                   ▼
   [t1, t2, t3...]                                       { Speaker Vector: S_omar }
          │                                                          │
          └───────────────────────────┬──────────────────────────────┘
                                      │ (Conditioning)
                                      ▼
                          ┌──────────────────────┐
                          │ Qwen3 LM (Backbone)  │
                          └──────────────────────┘
                                      │
                                      ▼ (Predicts C0_t)
                           { Semantic Token: C0_t }
                                      │
                                      ▼
                          ┌──────────────────────┐
                          │      MTP Module      │
                          └──────────────────────┘
                                      │
                                      ▼ (Predicts C1_t .. C15_t)
                   { Full Frame_t: [C0, C1, C2, ..., C15] }
                                      │
                                      ▼
                    ( Accumulate 4 Frames = 320ms Audio )
                                      │
                                      ▼
                      ┌──────────────────────────────┐
                      │ Streaming Causal ConvNet     │
                      │ Codec Decoder (12Hz)         │
                      └──────────────────────────────┘
                                      │
                                      ▼
                        (( Raw Waveform (Omar Voice) ))
========================================================================================================
```
-----------------------------------------------