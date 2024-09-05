Refer to README for Subgraph links

## Historical Swaps Query: 
```
{
  swaps(where: { pool: "<pool-id>" }, orderBy: timestamp, orderDirection: desc) {
    id
    timestamp
    amount0
    amount1
    sqrtPriceX96
    tick
    logIndex
  }
}
```
USDT/USDC Pool IDs:

0x3416cf6c708da44db2624d63ea0aaef7113527c6
- High liquidity/volume
- _data/historical-swaps/hs-USDT-USDC-high.txt_
  
0xbb256c2f1b677e27118b0345fd2b3894d2e6d487
- Low liquidity/volume
- _data/historical-swaps/hs-USDT-USDC-low.txt_

ETH/USDC Pool IDs:

0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- High volume/transaction count
- _data/historical-swaps/hs-ETH-USDC-high.txt_
  
0xe0554a476a092703abdb3ef35c80e0d76d32939f
- Low liquidity/volume
- _data/historical-swaps/hs-ETH-USDC-low.txt_

## Daily Pool Data Query: 
```
{
  poolDayDatas(where: { pool: "<pool-id>" }, orderBy: date, orderDirection: desc) {
    date
    liquidity
    sqrtPrice
    token0Price
    token1Price
    volumeUSD
    feesUSD
    tvlUSD
    txCount
  }
}

```
USDT/USDC Pool IDs:

0x3416cf6c708da44db2624d63ea0aaef7113527c6
- High liquidity/volume
- _data/daily-pool/dp-USDT-USDC-high.txt_
  
0xbb256c2f1b677e27118b0345fd2b3894d2e6d487
- Low liquidity/volume
- _data/daily-pool/dp-USDT-USDC-low.txt_

ETH/USDC Pool IDs:

0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- High volume/transaction count
- _data/daily-pool/dp-ETH-USDC-high.txt_
  
0xe0554a476a092703abdb3ef35c80e0d76d32939f
- Low liquidity/volume
- _data/daily-pool/dp-ETH-USDC-low.txt_

## Hourly Pool Data Query: 
```
{
  poolHourDatas(where: { pool: "<pool-id>" }, orderBy: periodStartUnix, orderDirection: desc) {
    periodStartUnix
    liquidity
    sqrtPrice
    token0Price
    token1Price
    volumeUSD
    feesUSD
    tvlUSD
    txCount
  }
}
```
USDT/USDC Pool IDs:

0x3416cf6c708da44db2624d63ea0aaef7113527c6
- High liquidity/volume
- _data/hourly-pool/hp-USDT-USDC-high.txt_
  
0xbb256c2f1b677e27118b0345fd2b3894d2e6d487
- Low liquidity/volume
- _data/hourly-pool/hp-USDT-USDC-low.txt_

ETH/USDC Pool IDs:

0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- High volume/transaction count
- _data/hourly-pool/hp-ETH-USDC-high.txt_
  
0xe0554a476a092703abdb3ef35c80e0d76d32939f
- Low liquidity/volume
- _data/hourly-pool/hp-ETH-USDC-low.txt_

## Mint and Burn Events Data Query: 
```
{
  mints(where: { pool: "<pool-id>" }, orderBy: timestamp, orderDirection: desc) {
    id
    timestamp
    amount
    amount0
    amount1
    amountUSD
  }
}
```
USDT/USDC Pool IDs:

0x3416cf6c708da44db2624d63ea0aaef7113527c6
- High liquidity/volume
- _data/mint/m-USDT-USDC-high.txt_
  
0xbb256c2f1b677e27118b0345fd2b3894d2e6d487
- Low liquidity/volume
- _data/mint/m-USDT-USDC-low.txt_

ETH/USDC Pool IDs:

0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- High volume/transaction count
- _data/mint/m-ETH-USDC-high.txt_
  
0xe0554a476a092703abdb3ef35c80e0d76d32939f
- Low liquidity/volume
- _data/mint/m-ETH-USDC-low.txt_

```
{
  burns(where: { pool: "<pool-id>" }, orderBy: timestamp, orderDirection: desc) {
    id
    timestamp
    amount
    amount0
    amount1
    amountUSD
  }
}
```
USDT/USDC Pool IDs:

0x3416cf6c708da44db2624d63ea0aaef7113527c6
- High liquidity/volume
- _data/burn/b-USDT-USDC-high.txt_
  
0xbb256c2f1b677e27118b0345fd2b3894d2e6d487
- Low liquidity/volume
- _data/burn/b-USDT-USDC-low.txt_

ETH/USDC Pool IDs:

0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
- High volume/transaction count
- _data/burn/b-ETH-USDC-high.txt_
  
0xe0554a476a092703abdb3ef35c80e0d76d32939f
- Low liquidity/volume
- _data/burn/b-ETH-USDC-low.txt_
