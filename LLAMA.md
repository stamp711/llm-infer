# Llama Model Architecture

## Model Configuration

### Core Parameters
- **Architecture**: Llama
- **Context Length**: 32,768 tokens
- **Embedding Dimensions**: 4,096
- **Transformer Blocks**: 32
- **Vocabulary Size**: 32,768 tokens

### Attention Mechanism
- **Attention Heads**: 32
- **Key-Value Heads**: 8
- **Head Dimension**: 128
- **Attention Type**: Grouped Query Attention (GQA)
  - 4:1 ratio of query heads to KV heads
  - Multiple query heads share key-value pairs for efficiency

### Position Encoding
- **Method**: Rotary Position Embeddings (RoPE)
- **RoPE Dimensions**: 128
- **RoPE Frequency Base**: 1,000,000

### Feed-Forward Network
- **FFN Hidden Size**: 14,336
- **Expansion Ratio**: ~3.5x embedding dimensions

### Normalization
- **Type**: RMS Normalization
- **Epsilon**: 1e-05

## Model Format
- **Quantization**: FP16 (16-bit floating point)
- **Tensor Count**: 291
- **Alignment**: 32 bytes

## Layer Structure

Each transformer block contains:
1. **Attention Layer**
   - Query, Key, Value projections
   - Multi-head attention with GQA
   - Output projection
2. **Feed-Forward Network**
   - Two linear transformations with activation
   - Expanded intermediate dimension (14,336)
3. **Normalization Layers**
   - RMS normalization before attention and FFN

## Processing Flow

1. **Input Embedding**: Tokens → 4,096-dimensional vectors
2. **Position Encoding**: RoPE applied to query and key vectors
3. **Transformer Blocks**: 32 sequential blocks process representations
4. **Output**: Final hidden states for next token prediction