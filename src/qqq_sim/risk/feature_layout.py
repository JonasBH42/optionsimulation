"""Immutable numerical feature layout shared by training and inference."""
FEATURE_SCHEMA = "EXACT_TOKENS_ACCOUNT_KINK_SHOCK_ACTION_V2"
TOKEN_FIELDS = ("mask", "log_moneyness", "remaining_years", "quantity_div_20", "right_sign", "iv", "multiplier_div_100", "half_spread_div_spot", "tick_div_spot", "american", "last_trade_years", "fixing_years")
ACTION_FAMILIES = ("HOLD", "FULL_CLOSE", "EXIT_STOP", "PARTIAL_CLOSE", "HEDGE", "REPOSITION", "ROLL_EXPIRATION")
ACCOUNT_FIELDS = ("economic_cash", "settled_cash", "unsettled_cash", "collateral", "working_order_reserve", "available_option_bp", "project_bp_used", "equity", "account_economic_pnl", "deficit", "bp_cap", "loss_cap", "session_loss_cap", "campaign_loss_bound", "session_loss_bound", "other_campaign_lower_bound", "campaign_available_cash", "campaign_remaining_loss_budget", "session_remaining_loss_budget")
CAMPAIGN_FIELDS = ("initial_deadline_years", "active_deadline_years", "hard_cutoff_years", "initial_fill_years", "remaining_actions_div_12", "remaining_rolls_div_3", "adjustment_count_div_12", "initial_fill_known", "stopped", "economically_completed", "unresolved_stock", "protection_recorded")
ACTION_FIELDS = ("premium_cost_div_A", "fees_div_A", "eligible_years", "limit_cost_div_A", "max_fees_div_A", "original_target_div_A", "stop_after_fill")
ORDER_TYPES = ("PROTECTED_LIMIT", "LIMIT", "MARKET")
TIME_IN_FORCE = ("DAY", "GTC")
SHOCK_MASK_OFFSET = 786
SHOCK_MASK_COUNT = 130
KINK_OFFSET = SHOCK_MASK_OFFSET + SHOCK_MASK_COUNT
HOLD_FEATURE_COUNT = KINK_OFFSET + 8*4 + 2
ACCOUNT_MASK_OFFSET = HOLD_FEATURE_COUNT + len(ACCOUNT_FIELDS)
CAMPAIGN_OFFSET = HOLD_FEATURE_COUNT + 2*len(ACCOUNT_FIELDS)
MANAGED_STATE_COUNT = CAMPAIGN_OFFSET + len(CAMPAIGN_FIELDS)
ACTION_SCALAR_OFFSET = MANAGED_STATE_COUNT + len(ACTION_FAMILIES)
ACTION_TOKEN_OFFSET = ACTION_SCALAR_OFFSET + len(ACTION_FIELDS) + len(ORDER_TYPES) + len(TIME_IN_FORCE)
UNIVERSE_OFFSET = ACTION_TOKEN_OFFSET + 16*len(TOKEN_FIELDS)
MANAGED_FEATURE_COUNT = UNIVERSE_OFFSET + 154
# A finite, explicitly masked encoding placeholder, never a computed price or
# a zero shock response. The matching missingness regime must be validated.
MISSING_SHOCK_SENTINEL = -9.87654321


def hold_masks():
    indices=[]
    for index in range(8):
        indices.extend(12+12*index+field for field in (0,4,9))
    for index in range(32):
        indices.extend(498+6*index+field for field in (0,5))
        indices.extend(690+3*index+field for field in (0,2))
    indices.extend(range(SHOCK_MASK_OFFSET,KINK_OFFSET))
    indices.extend(KINK_OFFSET+4*index for index in range(8))
    return tuple(sorted(indices))


def managed_masks():
    return tuple(sorted((*hold_masks(),*range(ACCOUNT_MASK_OFFSET,CAMPAIGN_OFFSET),
        *range(CAMPAIGN_OFFSET+7,MANAGED_STATE_COUNT),
        *range(MANAGED_STATE_COUNT,ACTION_SCALAR_OFFSET),ACTION_SCALAR_OFFSET+6,
        *range(ACTION_SCALAR_OFFSET+len(ACTION_FIELDS),ACTION_TOKEN_OFFSET),
        *(ACTION_TOKEN_OFFSET+12*index+field for index in range(16) for field in (0,4,9)),
        *(UNIVERSE_OFFSET+130+3*index for index in range(8)))))


def feature_contract():
    return {"schema":FEATURE_SCHEMA,"dtype":"float64","scale":"A=max(original_target_usd,loss_cap_usd,1)",
        "time_unit":"actual_seconds/31536000","hold_count":HOLD_FEATURE_COUNT,"managed_state_count":MANAGED_STATE_COUNT,
        "managed_count":MANAGED_FEATURE_COUNT,"position_tokens":{"count":8,"fields":TOKEN_FIELDS},
        "curves":{"nodes":65,"log_spot_range":["-log(2)","log(2)"],"order":["gross_mark","net_executable_liquidation","gross_terminal","net_terminal","iv_0.8_mark","iv_1.2_mark"]},
        "shock_masks":{"offset":SHOCK_MASK_OFFSET,"count":SHOCK_MASK_COUNT,"one_means":"UNAVAILABLE","placeholder":MISSING_SHOCK_SENTINEL,"admission":"EXACT_MISSINGNESS_PATTERN_VALIDATED_AS_SEPARATE_REGIME"},
        "kinks":{"count":8,"fields":["mask","log_strike_div_spot","gross_terminal_div_A","net_terminal_div_A"]},
        "target_gaps":["gross_current_mark_minus_H0_div_A","net_current_executable_liquidation_minus_H0_div_A"],
        "account_fields":ACCOUNT_FIELDS,"account_availability_masks":True,"campaign_fields":CAMPAIGN_FIELDS,
        "action_families":ACTION_FAMILIES,"action_fields":ACTION_FIELDS,"order_types":ORDER_TYPES,"time_in_force":TIME_IN_FORCE,
        "action_tokens":{"count":16,"fields":TOKEN_FIELDS},"universe_encoded_count":154,
        "hold_unscaled_indices":hold_masks(),"managed_unscaled_indices":managed_masks()}
