## Project Updates
#### Representing Ticks/Ranges
- v1: historical price range / N-price-intervals 
- v2: two-balance approach
- current: set of ticks/ranges (lowerTick, upperTick), each tick corresponds to a specific price: price(tick) = 1.0001^tick

#### Bonding-Curve Slippage Math
- v1: slippage as a %
- TODO: simplified version of Uniswap v3 formula: sqrt(p') = sqrt(p) + delta(x)/L
- TODO: track amount0, amount1 in the pool position, each swap changes these, range checks, separate logic for mint v. burn

#### Handling In-Range v. Out-of-Range Liquidity

#### Advanced Gas Fee

#### Expanded Environment State for Impermanent Loss

