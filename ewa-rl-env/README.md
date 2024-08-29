## Pool ID Selection:

Check token contract: https://etherscan.io/

Uniswap v3 Subgraph: https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV?view=Query&chain=arbitrum-one

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

