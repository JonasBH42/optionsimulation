# Explicit lifecycle inputs

`AppConfig.parameters_path` may point to a read-only JSON object containing both numerical scenario inputs and `lifecycle_events`. These records are runtime evidence or deliberately selected scenarios. The simulator does not derive assignments, official closes, financing rates or stock quotes from option candles.

Every record has a unique `id`, a supported `kind`, and offset-aware `economic_at` and `known_at` timestamps. Processing occurs at the later of those two times. An assignment notice with a later knowledge time cannot affect an earlier trading decision. Its economic effective time is retained for basis and dividend entitlement revisions. The sidecar's byte hash is frozen in saved session state; changing the file requires an explicit input-selection branch.

```json
{
  "rate_cc": 0.04,
  "dividend_yield_cc": 0.01,
  "sigma_path": 0.25,
  "known_at": "2026-11-09T09:30:00-05:00",
  "fixing_at": "2026-11-13T16:00:00-05:00",
  "source": "EXPLICIT_SYNTHETIC_SCENARIO",
  "lifecycle_events": [
    {
      "id": "example-fixing",
      "kind": "OFFICIAL_FIXING",
      "economic_at": "2026-11-13T16:00:00-05:00",
      "known_at": "2026-11-13T16:00:00-05:00",
      "expiry": "2026-11-13",
      "price": "597.00",
      "basis": "SYNTHETIC_TEST_FIXTURE"
    }
  ]
}
```

The prices and times above are synthetic examples. Supplying this example as an actual historical record would mislabel invented data.

| Kind | Additional fields | Effect |
| --- | --- | --- |
| `OFFICIAL_FIXING` | `price`, optional `expiry` (economic date by default), `basis` | Records a supplied fixing observation. Eligible expiry processing follows that observation. |
| `ASSIGNMENT` | `campaign_id`, `contract` or `contract_key`, positive integer `quantity`, optional `due_date`, `batch_id` | Removes only affected short contracts, transfers signed shares/strike cash and basis, and charges grouped assignment fees. Delayed notice withdraws the simultaneous-delivery certificate. |
| `EXERCISE` | Same identity/quantity fields as assignment | Exercises owned long contracts. Other hedges remain. |
| `PAIRED_DELIVERY` | `campaign_id`, `short`/`short_key`, `long`/`long_key`, `quantity`, optional `due_date` | Requires the preselected standing simultaneous-delivery model. Short assignment and protective exercise commit atomically, including out-of-money protection. |
| `EXERCISE_INSTRUCTION` | `campaign_id`, `contract`/`contract_key`, `instruction` = `DNE` or `EXERCISE`, optional exact `cutoff` | Validates long ownership and the instruction cutoff. Short DNE is prohibited. Required standing protection cannot be withdrawn by a conflicting DNE. |
| `DIVIDEND` | `per_share`, `pay_date`, optional `campaign_id`, `dividend_type` = `STANDARD_CASH` | Accrues signed economic entitlement. Later notices may create explicit entitlement corrections. Payment transfers buckets without a second income posting. Special dividends are unsupported without another profile. |
| `EXPIRY` | Optional `scenario`, `proxy_close` | Normally consumes the eligible official-fixing record. `scenario: "CLOSE_PROXY_V1"` with an explicit `proxy_close` selects a labeled `RTH_PROXY` outcome. |
| `EXPENSE` | `campaign_id`, `amount`, optional `category`, `due_date` | Accrues an explicit known expense once, reducing the remaining future-expense budget. |
| `FINANCING` | `campaign_id`, `due_date`, `stock_mark` and `borrow_rate` if short shares exist | Applies the declared whole-balance debit-rate / ACT/360 scenario and supplied stock-borrow rate. Missing borrow inputs are errors, never zero rates. |
| `STOCK_QUOTE` | `bid`, `ask`, optional `symbol` (QQQ), `quote_id`, `bid_size`, `ask_size` | A genuine later explicit quote may fill a submitted assignment-stock reduction. A candle close is not substituted. A quote known more than 60 seconds after its event time cannot fill. |
| `RISK_REVIEW` | Optional `campaign_id`, `automatic`, `minimum_package_cash`, `max_total_fees` | Examines supported assignment scenarios and the selected certificate. An unsafe exposure pauses for instruction unless a bounded automatic intervention was selected before entry. A resulting close order still needs a later eligible quote and may remain unfilled. |

An exact contract object uses `symbol`, `right` (`C`/`P`), decimal-string `strike`, ISO `expiry`, standard `multiplier` 100 and `deliverable` `QQQ_STANDARD_100_SHARES`. Optional `last_trade_at`, `fixing_at`, `root`, `style`, `currency` and `metadata_version` must match the held contract. A `contract_key` copied from a frozen session is the simplest unambiguous alternative. Keys are local canonical identities, not asserted OCC symbols.

Missing official fixing produces `SETTLEMENT_REFERENCE_MISSING` at the affected expiry boundary. A long exercise that needs unsupported funding or would create unapproved short stock produces a funding/exposure review stop. Recorded short assignments are economic events and remain represented even if they create a deficit. The named simultaneous-delivery model is conditional on atomic netted delivery and timely protective exercise; it is not an unconditional real-world broker guarantee.

Use `stock_reduce` in headless commands to reduce assignment stock:

```json
{
  "action": "stock_reduce",
  "campaign_id": "campaign",
  "quantity": -100,
  "minimum_cash_per_share": "597.00",
  "max_total_fees": "50.00",
  "command_id": "reduce-100-shares"
}
```

This sells up to 100 already-owned shares at a minimum of $597 per share. To cover existing short stock, use positive quantity and a negative limit (for example `-597.00` for a maximum $597 purchase). New stock exposure and reversals are rejected. Actual stock fees, FIFO basis, share conservation, partial depth consumption, quote identity and cash settlement are preserved. Missing depth is a disclosed simulation assumption, not verified liquidity.

Financing accruals retain unrounded Decimal amounts within their monthly charge group and post only cumulative cent differences. The group state and recognized day identities survive restore, so small daily charges are neither lost to repeated rounding nor charged twice. `due_date` must be supplied explicitly. Debit financing is attributed once per account/day even when several campaign borrow records share that date.

Initial campaign benchmarks are frozen at the actual completed initialization event, including a canceled partial opening committed as the final subset. The frozen record contains original cash and fees, target and deadlines, the then-visible option/IV state, the last completed underlying candle, and model/rule identities. Qualified coherent inputs show `PENDING` until asynchronous inference finishes; unavailable inputs show `UNKNOWN` with a reason. Later replay timestamps never substitute for the original benchmark. A probability result is recorded through an idempotent state transaction and cannot overwrite an already recorded initial result.
