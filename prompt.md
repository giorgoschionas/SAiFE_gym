I want to implement a generalized Poisson process in arrival_models.py which I named it as PoissonLinearArrivalModel. The overall intensity will be a linear combination of the three intensities. In particular, I am giving you the mathematical equation
$$a = \max (a_0, a_1 + a_2 \cdot L + a_3 \cdot (Z-S)) $$

where I define the intensities $a_0, a_1, a_2, a_3$ in the constructor as  `intensity: np.ndarray = np.array([140.0, 140.0], [130, 130], [120, 120], [110, 110])`.

The variables $L,Z,S$ are the state variables. In particular $L$ is the liquidity, which is usually accessed by `self.current_state[:, LIQUIDITY_INDEX]`. $Z$ is the AMM price which is accessed by `self.current_state[:, AMM_PRICE_INDEX]` and $S$ is the Asset price which is accessed by `self.current_state[:, ASSET_PRICE_INDEX]`.
Note that `self.current_state` is defined in the parent class `StochasticProcessModel`

So, based on that, I want you to think and implement the `get_arrivals` function in `PoissonLinearArrivalModel`
