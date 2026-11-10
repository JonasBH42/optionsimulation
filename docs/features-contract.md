The frozen feature schema is `EXACT_TOKENS_ACCOUNT_KINK_SHOCK_ACTION_V2`. `risk/feature_layout.py` defines the field order, offsets, sizes and flags that bypass standardization. Pack manifests carry both its version and complete serialized contract; qualification rejects older or incompatible contracts before network evaluation.

| Encoding | Float64 values | Content |
| --- | ---: | --- |
| HOLD | 950 | Exact position tokens; six 65-point gross/net/terminal/IV-shock curves; session/report clocks; 130 shock masks; eight exact-strike kink records; gross and net current target gaps |
| Managed state | 1,000 | HOLD plus 19 numeric account values, 19 availability masks, and 12 campaign clock/counter/state values |
| Managed action | 1,365 | Managed state plus family, premium/fee/limit/fee-cap/target/eligibility/stop fields, order-type and TIF flags, 16 exact change-leg tokens, and 154 learned universe features |

Dollar payoffs, cash, risk and action constraints use `A=max(original_target_usd, loss_cap_usd, 1)`. Time features use actual elapsed seconds divided by 31,536,000. Each option token retains its own economic fixing and exclusive trading stop. Kink records retain the gross and net cash bases separately. Current net liquidation includes natural sides, ticks and close fees; the target remains the same original dollar objective.

Account evidence is read from a reconstructed authoritative broker state without mutating the supplied ledger. It includes economic/settled/unsettled cash, collateral, working reserves, buying power, equity/P&L, deficits, caps and campaign/session loss budgets. An unavailable number has an explicit availability mask; an observed zero remains known. Offline `ManagedState` construction and transitions refresh derived evidence from their actual current account context, including remaining loss budget. Malformed context raises `INVALID_ACCOUNT_CONTEXT`.

Unsupported IV-shock coordinates receive the finite encoding placeholder `-9.87654321` and an unavailable mask. They are not interpreted as zero prices or zero sensitivity. Missing current marks still fail. Corrupt packed prices, cross-face failures, changed pack identity and lookup-budget failures remain hard errors. The exact shock/account missingness pattern forms part of the admitted coverage regime; a regime validated only for complete observations cannot authorize a different missingness pattern.

Limits, maximum fees, stop-after-fill, order type and TIF have distinct action features even when signed legs are identical. Exact broker feasibility and realized transitions remain authoritative; feature vectors never replace those checks. The universe encoder retains its 512-by-16 tokens and shared 16-to-64-to-64 network with masked pooled output.

The focused feature/universe suite passed 32 tests, covering these semantic distinctions, actual account transitions without prior-state mutation, typed malformed input, legacy-contract rejection, cash/fee target gaps, shock failures and coverage abstention. These tests establish encoding behavior, not numerical probability calibration or production model qualification.
