# Positional Encoding
Positional encoding describes the location (or position) of an entity in a sequence (in general, in a structure) so that each position is assigned a unique representation.

Let's start by understanding it inside a transformer for NLP tasks. It can be absolute or relative.

> [!Why can't we just use an index?]
>  With big sequences, the **magnitude** for the index would grow very large, so a normalization is required; But if we could have **sequences of variable length**, then the normalization assumes **different value for each position**: we can't either use an index or a normalized index because of the magnitude explosion and the normalization inconsistency.

The solution is having a positional scheme where *each position is mapped to a vector*. The output of the PE layer is a matrix where we have on the rows the various elements of the sequence (if a sequence has length $n$, then we'll have $n$ rows from 0 to $n-1$). Each row individually represents the updated embedding with the positional information, obtained by summing up the original embedding vector with the positional encoding vector for that position in the sequence.
## Absolute Positional **Encoding**
What we are going to do is simply get the input embedded vector and sum it to the position vector:
$\vec{E} + \vec{P}$
There are big limitations:
* With absolute positional encoding it is needed to have the set of position decided ahead of time. If the embedding space is smaller than the position, than we won't be able to have a positional representation for that bigger position values.
* If the same token appears in different positions, than there are two different representation of the same token, making hard for the model to understand that it is the same token but just in different places.
## Relative - Frequency Based Positional Encoding
Define a mathematical function that given a position will give us back a vector that encodes the position of that vector. Higher positions will oscillate more frequently.

Here the limitation 1 is removed, the second is slightly solved.
### How it works
let $d$ be the model dimension (originally called $d_{model}$), $pos$ the position of the token ( $0 \leq pos \lt n$) in the sequence, $i = 0, \dots, \frac{d}{2}-1$ is the mapping for the column indexes
$PE_{(pos,2i)} = \sin(\frac{pos}{10000^{\frac{2i}{d}}}) \space\space PE_{(pos,2i+1)} = \cos(\frac{pos}{10000^{\frac{2i}{d}}})$ 

Even position corresponds to a sine function, odd positions corresponds to a cosine function.
$i$ goes to half of the embedding dimension because every $i$ value is used both for the odd and even dimension (sine and cosine).
10000 is just a scalar that can be any number, it is just the one chosen in [[Attention Is All You Need.pdf]].![[Pasted image 20261008182946.png]]
* We can imagine the matrix of PE being filled by two for loops cycling on $pos$ from $0$ to $n$ and $i$ from $0$ to $\frac{d}{2}-1$.
* The sine and cosine function keeps the values in a normalized range for -1 to 1. 
* The PE matrix $\in \mathbb{R}^{n \times d_{model}}$.
* Every column represents a different sinusoid function based on $pos$, so every position holds its unique encoding. every row is unique.
* Every row is a vector and therefore we can measure their similarity via dot product, so we know how much two positions are close.