def test_round_trip():
    L = 1_000_000
    P0 = 2000
    sqrtP0 = P0 ** 0.5
    dx = 10             # token0 in
    sqrtP1, dy, fee = simulate_swap(dx, sqrtP0, L)

    # Swap back the opposite direction should restore price (after fee lost)
    # Implement simulate_swap_exact_out for completeness and assert |P2-P0| < 1e-6