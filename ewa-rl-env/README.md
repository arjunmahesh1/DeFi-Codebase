Purpose: Adapt [Uniswap Liquidity Provision: An Online Learning Approach](https://arxiv.org/pdf/2302.00610) as a Markov Decision Process, which might deal with the non-stationary nature of markets better and provide insights for LPs.

## Run Project:

Installation:
```
git clone https://github.com/arjunmahesh1/DeFi-Codebase.git
cd ewa-rl-env
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```
cd src
```
```
python preprocess_data.py
```
```
python train.py
```
```
python eval.py
```

Check out /notebooks/analysis.ipynb for analysis
Eta, Episodes can be changed in src/train.py

Env created using (OpenAI Gym)[https://www.gymlibrary.dev/api/core/]

// TODO: add project explanation, documentation, checkpoints 



## Pool ID Selection:

Check token contract: https://etherscan.io/

Uniswap v3 Subgraph: https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV?view=Query&chain=arbitrum-one

Uniswap v3 Subgraph Schema: https://github.com/Uniswap/v3-subgraph/blob/main/schema.graphql#L75

Refer to /data/queries.md for pool subqueries

**USDT/USDC**

Query for pools:

```
{
  pools(where: { token0: "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", token1: "0xdac17f958d2ee523a2206206994597c13d831ec7" }) {
    id
    volumeUSD
    liquidity
    txCount
    token0 {
      symbol
      id
    }
    token1 {
      symbol
      id
    }
  }
}
```
Pool IDs Results:

0x3416cf6c708da44db2624d63ea0aaef7113527c6
- High liquidity/volume
  
0xbb256c2f1b677e27118b0345fd2b3894d2e6d487
- Low liquidity/volume



**ETH/USDC**

Query for pools:

```
{
  pools(where: { token0: "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", token1: "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2" }) {
    id
    volumeUSD
    liquidity
    txCount
    token0 {
      symbol
      id
    }
    token1 {
      symbol
      id
    }
  }
}
```
Pool IDs Results:

0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- High volume/transaction count
  
0xe0554a476a092703abdb3ef35c80e0d76d32939f
- Low liquidity/volume


// TODO: Make dynamic pool querying, integrate API key/wallet 

